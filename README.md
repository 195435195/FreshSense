# FreshSense 食品无损检测

面向研究展示的 Streamlit 原型，具备模拟/CSV 数据接入、传感器信号校验、自动判定、历史记录和 CSV 报告导出。

## 在 IDEA 中运行

1. 用 IDEA 打开本文件夹。
2. 在 Terminal 执行 `pip install -r requirements.txt`。
3. 执行 `streamlit run app.py`，再打开终端输出的本地网址。

## CSV 格式

至少包括一列时间与四列传感器信号：

```csv
time_s,S01,S02,S03,S04
0,0.001,0.002,-0.001,0.000
2,0.005,0.003,-0.002,0.004
```

`time_s` 也可命名为 `时间 (s)`。传感器列需以 `S` 开头，如 `S01` 至 `S16`。
