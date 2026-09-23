# 实验工具与历史结果

这里将原 `jax-stack` 的探针、验证器、补丁、JSON 结果和辅助代码放在一起，以保留 Python 模块导入与 `HERE` 相对路径依赖。典型入口包括 [源码索引生成](build_source_index.py)、[CPU matmul capture](matmul_probe.py)、[Pallas 对照](pallas_probe.py) 和 [源码索引渲染](render_source_index.py)。按主题阅读的说明文档从 [上级索引](../README.md)进入。

JSON、Notebook 的已执行输出及旧 capture 路径是历史记录，未随目录重排重写其内部 provenance。要复现或重新验证，应先核对锁文件、运行时身份与上游源码材料；旧路径、hash 或环境缺失不能被解释为新结果。
