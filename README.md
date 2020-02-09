<div align=center>
    <div style="font-size:24px">
        <b>Four Factor Framework For Fact Finding</b>
    </div>
</div>

The First Workshop on Fact Extraction and Verification


## Requirement
    * Python 3.7.4
    * torch 1.2.0
    * tqdm
    * nltk
    * jack
    * pytz
    * fever-score
    * 运行环境：1080 Ti GPU CUDA 10.0


## Prepare

1. 安装以上需要的所有模块.

2.  在 fever 同一目录下安装 jack 框架: 
    ```bash
    git clone https://github.com/takuma-ynd/jack.git

    # 安装 jack框架需要的模块，并下载预训练的 GloVe 数据
    cd jack
    python -m pip install -e .[tf]
    bash ./data/GloVe/download.sh
    ```
    注：准备工作仅需要完成一次即可. 

3. fever 项目目录结构如下：

```python
├── jack                                    # jack 框架目录
│
└── fever                                   # fever 项目目录
    ├── data                                # 数据相关文件
    │   ├── dataset                         # FEVER 数据集
    │   │   ├── train.jsonl                 # 训练集
    │   │   ├── dev.jsonl                   # 开发集
    │   │   └── test.jsonl                  # 测试集
    │   │
    │   ├── preprocessed_data               # 预处理后的数据
    │   │   ├── doctitle                    # 文档标题字典
    │   │   └── edocs.bin
    │   │
    │   ├── reader                          # jack 框架下预训练的 ESIM 模型
    │   │
    │   ├── wiki-pages                      # 维基百科文档数据
    │   │
    │   └── stoplist                        # 停用词表
    │
    ├── results                             # 模型运行时各模块的输出结果
    │   ├── doc_ret                         # 文档检索结果
    │   ├── sent_ret                        # 语句检索结果
    │   ├── evi_ret                         # 证据检索结果
    │   ├── nli                             # 自然语言推断结果
    │   ├── aggregator                      # 整合预测结果
    │   ├── rerank                          # 重新排序结果
    │   └── score                           # 评估结果
    │
    ├── run_model.py                        # 运行 FEVER 系统模型
    ├── document_retrieval.py               # 文档检索
    ├── sentence_retrieval.py               # 语句检索
    ├── evidence_retrieval.py               # 在检索的语句中提取证据
    ├── natural_language_inference.py       # 自然语言推断
    ├── natural_aggregator.py               # 整合预测
    ├── rerank.py                           # 对预测结果重新排序
    ├── score.py                            # 对预测结果进行评估
    ├── fever_io.py                         # 数据处理及读写操作
    ├── analyse.py                          # 结果评估
    ├── util.py                             # 数据结构定义
    └── configs.py                          # 系统整体参数配置
```

4. 根据运行环境修改 config.py 文件中的所有的路径参数


## Train

* 重新训练 fever 系统：

1. 删除 
    ```bash
    ./data/preprocessed_data/
    ```
    目录下的所有文件

2. 删除
    ```bash
    ./results/
    ```
    目录下的所有文件夹

3. 运行
    ```bash
    python run_model.py
    ```


## Reimplement the Result

* 对最终实现的 FEVER Score 70.27% 进行复现：

1. 删除文件夹
    ```bash
    ./results/score
    ```

2. 运行
    ```bash
    python run_model.py
    ```

## Original Paper
[UCL Machine Reading Group:
Four Factor Framework For Fact Finding (HexaF)](http://aclweb.org/anthology/W18-5515)
