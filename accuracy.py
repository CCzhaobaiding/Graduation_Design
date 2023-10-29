import re
import pandas as pd

log_file = "log/cifar10@40fixmatch.log"
data = []

# 使用正则表达式来匹配需要的数据行
pattern = r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} - INFO - __main__ -   top-1 acc: (\d+\.\d+)'

with open(log_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

for line in lines:
    match = re.search(pattern, line)
    if match:
        acc = float(match.group(1))
        data.append(acc)

# 只选择前200次的数据
data = data[:200]

# 将数据存储到Excel文件
df = pd.DataFrame({"Top-1 Accuracy": data})
df.to_excel("top1_accuracy.xlsx", index=False)

print("数据已保存到top1_accuracy.xlsx文件")
