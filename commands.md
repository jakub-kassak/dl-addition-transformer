# Commands

## Upload code to cluster
```bash
rsync -avzu --exclude-from='.rsync-exclude' . student-cluster:addition-transformer/
```

## Train on cluster
```bash
sbatch submit_train.sh
```

## Download artifacts
```bash
rsync -avzu student-cluster:addition-transformer/logs/ logs/ && rsync -avzu student-cluster:addition-transformer/experiments/ experiments/
```