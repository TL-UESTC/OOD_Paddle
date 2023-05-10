import paddle
from paddle.io import Dataset, DataLoader


class Dataset(paddle.io.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()


class DataLoader(paddle.io.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False,
               sampler=None, batch_sampler=None,
               num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False,
               timeout=0, worker_init_fn=None,
               multiprocessing_context=None,
               generator=None, *, prefetch_factor=2,
               persistent_workers=False, pin_memory_device=''):
        super(DataLoader, self).__init__(dataset=dataset, feed_list=None,
                                        places=None, return_list=True,
                                        batch_sampler=batch_sampler, batch_size=batch_size, 
                                        shuffle=shuffle, drop_last=drop_last,
                                        collate_fn=collate_fn, num_workers=num_workers,
                                        use_buffer_reader=True, use_shared_memory=True,
                                        timeout=timeout, worker_init_fn=worker_init_fn)