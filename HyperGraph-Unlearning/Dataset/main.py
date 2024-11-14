import os
from dhg.data import Cooking200, CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer


def download_datasets():
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 数据集列表和对应的类
    datasets = {
        "Cooking200": Cooking200,
        "CoauthorshipCora": CoauthorshipCora,
        "CoauthorshipDBLP": CoauthorshipDBLP,
        "CocitationCora": CocitationCora,
        "CocitationCiteseer": CocitationCiteseer
    }

    # 下载每个数据集并存储到各自的文件夹中
    for name, dataset_class in datasets.items():
        # 创建以数据集名称命名的文件夹
        dataset_dir = os.path.join(current_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 初始化数据集实例并指定下载路径
        dataset = dataset_class(data_root=dataset_dir)

        # 访问数据集的常规属性以触发下载
        _ = dataset["num_vertices"]
        _ = dataset["num_classes"]
        _ = dataset["edge_list"]
        _ = dataset["labels"]
        _ = dataset["train_mask"]
        _ = dataset["val_mask"]
        _ = dataset["test_mask"]

        # 检查并访问可选的属性
        if "dim_features" in dataset._content:
            _ = dataset["dim_features"]
        if "features" in dataset._content:
            _ = dataset["features"]

        print(f"{name} dataset files have been downloaded successfully to {dataset_dir}.")


if __name__ == "__main__":
    download_datasets()
