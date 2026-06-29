import os
import re
import pandas as pd

log_folder = "./log"

# 创建一个空的DataFrame来保存数据
all_data = []

# 递归遍历文件夹及其子文件夹
for root, dirs, files in os.wal我k(log_folder):
    for log_file_name in files:
        if log_file_name.endswith(".log"):
            log_file_path = os.path.join(root, log_file_name)

            with open(log_file_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    log_line = line.strip()
                    match = re.search(r'Epoch (\d+): train_loss=(.*?), pre=(.*?), recall=(.*?), dice=(.*?), jaccard=(.*?),spe=(.*?),acc=(.*?),f1=(.*?),test_loss=(.*?),', log_line)
                    if match:
                        epoch, train_loss, pre, recall, dice, jaccard, spe, acc, f1, test_loss = match.groups()
                        data = {
                            'Epoch': int(epoch),
                            'Train Loss': float(train_loss),
                            'Precision': float(pre),
                            'Recall': float(recall),
                            'Dice': float(dice),
                            'Jaccard': float(jaccard),
                            'Sensitivity': float(spe),
                            'Accuracy': float(acc),
                            'F1 Score': float(f1),
                            'Test Loss': float(test_loss)
                        }
                        all_data.append(data)

            # 将数据转换为DataFrame
            df = pd.DataFrame(all_data)
            # 保存DataFrame为.csv文件
            df.to_csv(log_file_path.replace(".log", ".csv"), index=None)
