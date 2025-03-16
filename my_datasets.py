import torch
from torch.utils.data import Dataset
import h5py
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder=None, split=None, transform=None, ecgs=None, captions=None, caplens=None, read_from_hdf5=True):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        if read_from_hdf5:
            self.split = split
            assert self.split in {'TRAIN', 'VAL', 'TEST'}

            # Open hdf5 file where images are stored
            self.h = h5py.File(os.path.join(data_folder, self.split + '_DATASET.hdf5'), 'r')
            self.secgs = self.h['secgs']
            self.captions = self.h['captions']
            self.caplens = self.h['caplens']
        else:
            self.secgs = ecgs
            self.captions = captions
            self.caplens = caplens

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


    def __getitem__(self, i):
        secg = self.secgs[i]
        if not isinstance(secg, torch.Tensor):
            secg = torch.FloatTensor(secg)
        if self.transform is not None:
            secg = self.transform(secg)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([round(self.caplens[i])])

        return secg, caption, caplen

    def __len__(self):
        return self.dataset_size


def load_and_split_data(data_folder, train_csv, val_csv, test_csv):
    """reads an h5py file and split it to train/val/test according to csv files"""

class TransformerDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_DATASET.hdf5'), 'r')
        self.secgs = self.h['secgs'][:]
        self.captions = self.h['captions'][:]
        self.caplens = self.h['caplens'][:]

        self.dataset_size = len(self.captions)


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        secg = torch.FloatTensor(self.secgs[i])
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([round(self.caplens[i])])

        x = (secg, caption[:-1])
        y = (caption)

        return x, y, caplen
       

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    ds = TransformerDataset('data/not_balanced/not_deduplicated/commented/BPE_encoded/', 'DATASET', 'TRAIN')
    x = next(iter(ds))
    print()


class SECGDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_ASSESSMENT.hdf5'), 'r')
        self.secgs = self.h['secgs']
        # assessment per sECG
        self.assessments = self.h['assessments']


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # # Total number of datapoints
        self.dataset_size = len(self.secgs)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        secg = torch.FloatTensor(self.secgs[i])
        assessment = torch.FloatTensor(self.assessments[i])
        if self.transform is not None:
            secg = torch.FloatTensor([self.secgs[i]])
            secg = self.transform(secg)[0]

        return secg, assessment

    def __len__(self):
        return self.dataset_size
