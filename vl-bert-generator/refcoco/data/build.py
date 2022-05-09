import torch.utils.data
import os

from .datasets import *
from . import samplers
from .transforms.build import build_transforms
from .collate_batch import BatchCollator
import pprint


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size):
    if aspect_grouping:
        group_ids = dataset.group_ids
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False
        )
    return batch_sampler


def make_dataloader(cfg, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None,
                    expose_sampler=False):
    assert mode in ['train', 'val', 'test']
    sketch_root_path = os.path.join(cfg.DATASET.SYNZ_ROOT, 'images')
    sketch_root_path = os.path.join(sketch_root_path, mode)
    prediction_path = os.path.join(cfg.DATASET.PREDICTION_ROOT, mode + '.pth')
    description_path = cfg.DATASET.DESCRIPTION_PATH
    synz_annotation_path = os.path.join(cfg.DATASET.SYNZ_ROOT, 'annotations/' + mode +'.json')
    if mode == 'train':
        # ann_file = cfg.DATASET.TRAIN_ANNOTATION_FILE
        # image_set = cfg.DATASET.TRAIN_IMAGE_SET
        aspect_grouping = cfg.TRAIN.ASPECT_GROUPING
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        # boxes = cfg.DATASET.TRAIN_BOXES
    elif mode == 'val':
        # ann_file = cfg.DATASET.VAL_ANNOTATION_FILE
        # image_set = cfg.DATASET.VAL_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        # boxes = cfg.DATASET.VAL_BOXES
    else:
        # ann_file = cfg.DATASET.TEST_ANNOTATION_FILE
        # image_set = cfg.DATASET.TEST_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        # boxes = cfg.DATASET.TEST_BOXES

    transform = build_transforms(cfg, mode)

    if dataset is None:
        
        dataset = SynZ(sketch_root_path=sketch_root_path, prediction_path=prediction_path, 
                       description_path=description_path, synz_annotations_path=synz_annotation_path,
                       pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME, transform=transform,
                       add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
                       aspect_grouping=aspect_grouping, test_mode=(mode=='test'))

        # dataset = build_dataset(dataset_name=cfg.DATASET.DATASET, ann_file=ann_file, image_set=image_set,
        #                         boxes=boxes, proposal_source=cfg.DATASET.PROPOSAL_SOURCE,
        #                         answer_vocab_file=cfg.DATASET.ANSWER_VOCAB_FILE,
        #                         root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
        #                         test_mode=(mode == 'test'), transform=transform,
        #                         zip_mode=cfg.DATASET.ZIP_MODE, cache_mode=cfg.DATASET.CACHE_MODE,
        #                         cache_db=True if (rank is None or rank == 0) else False,
        #                         ignore_db_cache=cfg.DATASET.IGNORE_DB_CACHE,
        #                         add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
        #                         aspect_grouping=aspect_grouping,
        #                         pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME)

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size)
    collator = BatchCollator(dataset=dataset, append_ind=cfg.DATASET.APPEND_INDEX)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collator)
    if expose_sampler:
        return dataloader, sampler

    return dataloader
