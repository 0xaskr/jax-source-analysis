# JAX 内部软件栈

这一目录从 JAX 自身的概念与实现出发，不固定某个模型、算子或 kernel。当前选定的组件主图为 [Jaxpr 中心辐射图](jaxpr-centered-hub.svg)，沿用 overview 分层图的浅色大框、核心节点与流程连线，围绕 Jaxpr 展开定义、产生、变换与消费。

中央 Jaxpr 连接上方定义、左侧产生、下方变换与右侧消费。定义解释 Jaxpr 自身；产生路径明确输出 Jaxpr；变换路径明确输入与输出 Jaxpr；消费路径从 Jaxpr 指向值、文本、校验结果或 MLIR。蓝色框始终表示研究对象，生成机制、变换规则与消费者围绕它展开。

分区表示研究视角，各变换分支按需组合，不是所有调用必经的固定串行流水线。主图不包含模型示例或 call-to-LLO 跟踪，也不展开 IFRT / PJRT / XLA 的内部结构。

主图由 [`render_jaxpr_centered_hub.py`](../../../tools/render_jaxpr_centered_hub.py) 生成，复用当前固定源码锚点。运行 `python3 -B tools/render_jaxpr_centered_hub.py --preview-dir /tmp/jaxpr-centered-preview` 仅重建中心辐射版；添加 `--output-dir` 可先写入临时目录。脚本同时检查关系端点：产生必须指向 Jaxpr，变换必须以 Jaxpr 为输入和输出，消费必须从 Jaxpr 出发。

主图按 JAX `361c43e072cce92b7d3e9bdaf4dd16db26c49043` 核对，修订以 [`upstream-sources.lock`](../../../upstream-sources.lock) 和实际检出共同验证。图内链接指向固定提交的类、函数与字段；证据仅为源码检查和图示生成，不代表 CPU/GPU/TPU 执行。

此前的 [JAX 内部架构图](jax-internal-stack.svg) 保留为补充材料，其生成脚本 [`render_jax_internal_stack.py`](../../../tools/render_jax_internal_stack.py) 同时提供中心辐射版复用的源码锚点与版本核验；字体测量和绘图工具由 overview 生成器提供。中心辐射版已替代此前架构图作为当前组件主图。

已有文字材料包括 [概念与下游接口](jax-concepts.md) 和 [Jaxpr 的格式与打印语法](jaxpr-format.md)；其中旧版本记录与专题示例不作为本次组件主图的运行证据，引用时需核对相应源码版本。
