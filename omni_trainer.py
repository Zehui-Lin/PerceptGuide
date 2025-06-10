import os
import cv2
import sys
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import DiceLoss
from datasets.dataset import USdataset, USdatasetCls
from datasets.omni_dataset import WeightedRandomSamplerDDP
from datasets.omni_dataset import USdatasetOmni
from datasets.dataset import RandomGenerator, CenterCropGenerator
from sklearn.metrics import roc_auc_score
from utils import omni_seg_test


def weight_base_init(nn_dataset):
    from datasets.omni_dataset import position_prompt_dict

    position_num_dict = {}
    seg_use_dataset_num = len(nn_dataset.seg_use_dataset)
    # cls_use_dataset_num = len(nn_dataset.cls_use_dataset)
    for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
        if position_prompt_dict[dataset_name] not in position_num_dict:
            position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]
        else:
            position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
    for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
        if position_prompt_dict[dataset_name] not in position_num_dict:
            position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[
                seg_use_dataset_num + dataset_index
            ]
        else:
            position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[
                seg_use_dataset_num + dataset_index
            ]

    position_weight_dict = {}
    for position in position_num_dict:
        position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])

    all_sample_weight_list = []
    for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
        all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[
            dataset_index
        ]
    for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
        all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[
            seg_use_dataset_num + dataset_index
        ]

    return all_sample_weight_list


def omni_train(args, model, snapshot_path):
    # GPU Device
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200)
    )  # might takes a long time to sync between process
    # Display & Logging
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("** GPU NUM ** : ", torch.cuda.device_count())
        print("** WORLD SIZE ** : ", torch.distributed.get_world_size())
    print(f"** DDP ** : Start running on rank {rank}.")
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # lr, batch size
    base_lr = args.base_lr
    batch_size = args.batch_size

    # Data
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # omni dataset
    db_train = USdatasetOmni(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        prompt=True,
    )

    sample_weight_seq = weight_base_init(db_train)
    weighted_sampler = WeightedRandomSamplerDDP(
        data_set=db_train,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=len(sample_weight_seq),
        replacement=True,
    )
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=weighted_sampler,
    )

    # Model
    model = model.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    model.train()

    # Loss
    seg_ce_loss = CrossEntropyLoss()
    seg_dice_loss = DiceLoss()
    cls_ce_loss = CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # Resume
    resume_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
        optimizer.load_state_dict(torch.load(args.resume, map_location="cpu")["optimizer"])
        resume_epoch = torch.load(args.resume, map_location="cpu")["epoch"]

    # Logging
    writer = SummaryWriter(snapshot_path + "/log")
    best_performance = 0.0
    best_epoch = 0

    global_iter_num = 0

    total_iterations = len(trainloader)

    max_iterations = args.max_epochs * total_iterations
    logging.info(
        "{} batch size. {} iterations per epoch. {} max iterations ".format(
            batch_size, total_iterations, max_iterations
        )
    )

    # in ddp, only master process display the progress bar
    if int(os.environ["LOCAL_RANK"]) != 0:
        iterator = tqdm(range(resume_epoch, args.max_epochs), ncols=70, disable=True)
    else:
        iterator = tqdm(range(resume_epoch, args.max_epochs), ncols=70, disable=False)

    # Training Loop
    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))

        weighted_sampler.set_epoch(epoch_num)
        for i_batch, sampled_batch in tqdm(enumerate(trainloader)):

            image_batch = sampled_batch["image"]
            task_batch = sampled_batch["task"]
            label_batch1D = sampled_batch["1dLabel"]
            label_batch2D = sampled_batch["2dLabel"]

            image_batch = image_batch.to(device=device)
            label_batch1D = label_batch1D.to(device=device)
            label_batch2D = label_batch2D.to(device=device)

            if args.prompt:
                position_prompt = (
                    torch.tensor(np.array(sampled_batch["position_prompt"])).permute([1, 0]).float().to(device=device)
                )
                task_prompt = (
                    torch.tensor(np.array(sampled_batch["task_prompt"])).permute([1, 0]).float().to(device=device)
                )
                mode_prompt = (
                    torch.tensor(np.array(sampled_batch["mode_prompt"])).permute([1, 0]).float().to(device=device)
                )
                type_prompt = (
                    torch.tensor(np.array(sampled_batch["type_prompt"])).permute([1, 0]).float().to(device=device)
                )
                (x_seg, x_cls) = model(
                    (
                        image_batch,
                        position_prompt,
                        task_prompt,
                        mode_prompt,
                        type_prompt,
                    )
                )
            else:
                (x_seg, x_cls) = model(image_batch)

            x_seg_select_index = [
                task_batch[element_index] == "segmentation" for element_index, _ in enumerate(task_batch)
            ]
            x_cls_select_index = [
                task_batch[element_index] == "classification" for element_index, _ in enumerate(task_batch)
            ]

            # seg
            if sum(x_seg_select_index) == 0:
                loss_seg = 0
            else:
                loss_ce = seg_ce_loss(
                    x_seg[x_seg_select_index],
                    label_batch2D[x_seg_select_index].long(),
                )
                loss_dice = seg_dice_loss(
                    x_seg[x_seg_select_index],
                    label_batch2D[x_seg_select_index],
                    softmax=True,
                )
                loss_seg = 0.4 * loss_ce + 0.6 * loss_dice
            # cls
            if sum(x_cls_select_index) == 0:
                loss_cls = 0
            else:
                loss_cls = cls_ce_loss(
                    x_cls[x_cls_select_index],
                    label_batch1D[x_cls_select_index].long(),
                )

            loss = loss_seg + loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - global_iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            global_iter_num = global_iter_num + 1

            writer.add_scalar("info/lr", lr_, global_iter_num)
            writer.add_scalar("info/total_loss", loss, global_iter_num)
            logging.info("global iteration %d and loss : %f" % (global_iter_num, loss.item()))

        dist.barrier()

        if int(os.environ["LOCAL_RANK"]) == 0:
            torch.cuda.empty_cache()

            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_num,
            }
            save_latest_path = os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num))
            if os.path.exists(os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num - 1))):
                os.remove(os.path.join(snapshot_path, "latest_{}.pth".format(epoch_num - 1)))
                os.remove(os.path.join(snapshot_path, "latest.pth"))
            torch.save(save_dict, save_latest_path)
            os.system("ln -s " + os.path.abspath(save_latest_path) + " " + os.path.join(snapshot_path, "latest.pth"))

            model.eval()
            total_performance = 0.0

            # seg
            seg_val_set = [
                "DDTI",
                "MMOTU",
                "TN3K",
                "Fetal_HC",
                "BUSIS",
                "CCAU",
                "BUS-BRA",
                "kidneyUS_capsule",
                "EchoNet-Dynamic",
                "UDIAT",
            ]

            seg_avg_performance = 0.0

            for dataset_name in seg_val_set:
                num_classes = (
                    open(os.path.join(args.root_path, "segmentation", dataset_name, "config.yaml")).read().count("\n")
                )
                db_val = USdataset(
                    base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                    transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt,
                )
                val_loader = DataLoader(
                    db_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                logging.info("{} val iterations per epoch".format(len(val_loader)))

                metric_list = 0.0
                count_matrix = np.ones((len(db_val) * num_classes, num_classes - 1))
                for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt:
                        position_prompt = (
                            torch.tensor(np.array(sampled_batch["position_prompt"])).permute([1, 0]).float()
                        )
                        task_prompt = (
                            torch.tensor(
                                np.array(
                                    [
                                        [1] * position_prompt.shape[0],
                                        [0] * position_prompt.shape[0],
                                    ]
                                )
                            )
                            .permute([1, 0])
                            .float()
                        )
                        mode_prompt = torch.tensor(np.array(sampled_batch["mode_prompt"])).permute([1, 0]).float()
                        type_prompt = torch.tensor(np.array(sampled_batch["type_prompt"])).permute([1, 0]).float()
                        metric_i = omni_seg_test(
                            image,
                            label,
                            model,
                            classes=num_classes,
                            prompt=args.prompt,
                            position_prompt=position_prompt,
                            task_prompt=task_prompt,
                            mode_prompt=mode_prompt,
                            type_prompt=type_prompt,
                        )
                    else:
                        metric_i = omni_seg_test(image, label, model, classes=num_classes)

                    for sample_index in range(len(metric_i)):
                        if not metric_i[sample_index][1]:
                            count_matrix[
                                (i_batch * batch_size + sample_index) // (num_classes - 1),
                                sample_index % (num_classes - 1),
                            ] = 0
                    metric_i = [element[0] for element in metric_i]
                    metric_list += np.array(metric_i).sum()

                performance = metric_list / (count_matrix.sum() + 1e-6)

                writer.add_scalar(
                    "info/val_seg_metric_{}".format(dataset_name),
                    performance,
                    epoch_num,
                )

                seg_avg_performance += performance

            seg_avg_performance = seg_avg_performance / (len(seg_val_set) + 1e-6)
            total_performance += seg_avg_performance
            writer.add_scalar("info/val_metric_seg_Total", seg_avg_performance, epoch_num)

            # cls
            cls_val_set = [
                "TN3K",
                "CUBS",
                "BUS-BRA",
                "Appendix",
                "Fatty-Liver",
                "UDIAT",
            ]

            cls_avg_performance = 0.0

            for dataset_name in cls_val_set:
                num_classes = (
                    open(os.path.join(args.root_path, "classification", dataset_name, "config.yaml")).read().count("\n")
                )
                db_val = USdatasetCls(
                    base_dir=os.path.join(args.root_path, "classification", dataset_name),
                    split="val",
                    list_dir=os.path.join(args.root_path, "classification", dataset_name),
                    transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                    prompt=args.prompt,
                )

                val_loader = DataLoader(
                    db_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                logging.info("{} val iterations per epoch".format(len(val_loader)))
                model.eval()

                label_list = []
                prediction_prob_list = []
                for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    if args.prompt:
                        position_prompt = (
                            torch.tensor(np.array(sampled_batch["position_prompt"])).permute([1, 0]).float()
                        )
                        task_prompt = (
                            torch.tensor(
                                np.array(
                                    [
                                        [0] * position_prompt.shape[0],
                                        [1] * position_prompt.shape[0],
                                    ]
                                )
                            )
                            .permute([1, 0])
                            .float()
                        )
                        mode_prompt = torch.tensor(np.array(sampled_batch["mode_prompt"])).permute([1, 0]).float()
                        type_prompt = torch.tensor(np.array(sampled_batch["type_prompt"])).permute([1, 0]).float()
                        with torch.no_grad():
                            output = model(
                                (
                                    image.cuda(),
                                    position_prompt.cuda(),
                                    task_prompt.cuda(),
                                    mode_prompt.cuda(),
                                    type_prompt.cuda(),
                                )
                            )[1]
                    else:
                        with torch.no_grad():
                            output = model(image.cuda())[1]

                    out_label_back_transform = torch.cat([output[:, 0:1], output[:, 1:num_classes]], axis=1)
                    output_prob = torch.softmax(out_label_back_transform, dim=1).data.cpu().numpy()

                    label_list.append(label.numpy())
                    prediction_prob_list.append(output_prob)

                label_list = np.expand_dims(
                    np.concatenate(
                        (
                            np.array(label_list[:-1]).flatten(),
                            np.array(label_list[-1]).flatten(),
                        )
                    ),
                    axis=1,
                ).astype("uint8")
                label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                performance = roc_auc_score(
                    label_list_OneHot,
                    np.concatenate(
                        (
                            np.array(prediction_prob_list[:-1]).reshape(-1, num_classes),
                            prediction_prob_list[-1],
                        )
                    ),
                    multi_class="ovo",
                )

                writer.add_scalar(
                    "info/val_cls_metric_{}".format(dataset_name),
                    performance,
                    epoch_num,
                )

                cls_avg_performance += performance

            cls_avg_performance = cls_avg_performance / (len(cls_val_set) + 1e-6)
            total_performance += cls_avg_performance
            writer.add_scalar("info/val_metric_cls_Total", cls_avg_performance, epoch_num)

            TotalAvgPerformance = total_performance / 2

            logging.info("This epoch %d Validation performance: %f" % (epoch_num, TotalAvgPerformance))
            logging.info("But the best epoch is: %d and performance: %f" % (best_epoch, best_performance))
            writer.add_scalar("info/val_metric_TotalMean", TotalAvgPerformance, epoch_num)
            if TotalAvgPerformance >= best_performance:
                if os.path.exists(
                    os.path.join(
                        snapshot_path,
                        "best_model_{}_{}.pth".format(best_epoch, round(best_performance, 4)),
                    )
                ):
                    os.remove(
                        os.path.join(
                            snapshot_path,
                            "best_model_{}_{}.pth".format(best_epoch, round(best_performance, 4)),
                        )
                    )
                    os.remove(os.path.join(snapshot_path, "best_model.pth"))
                best_epoch = epoch_num
                best_performance = TotalAvgPerformance
                logging.info("Validation TotalAvgPerformance in best val model: %f" % (TotalAvgPerformance))
                save_model_path = os.path.join(
                    snapshot_path,
                    "best_model_{}_{}.pth".format(epoch_num, round(best_performance, 4)),
                )
                os.system(
                    "ln -s " + os.path.abspath(save_model_path) + " " + os.path.join(snapshot_path, "best_model.pth")
                )
                torch.save(model.state_dict(), save_model_path)
                logging.info("save model to {}".format(save_model_path))

        model.train()

    writer.close()
    return "Training Finished!"
