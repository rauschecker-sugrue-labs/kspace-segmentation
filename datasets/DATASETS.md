# Dataset Configuration

To use data like the OASIS-1 dataset, you have to create symbolic links in this directory, which point to the actual path of your stored dataset. 

```bash
ln -s /path/to/UPENN_GBM/ ./kspace-segmentation/datasets/UPENN_GBM
ln -s /path/to/OASIS/ ./kspace-segmentation/datasets/OASIS
```