from ..models import (
    ConfigFieldSpec,
    ConfigOptionSpec,
    Framework,
    InputPortSpec,
    NodeSpec,
    NodeTemplateSpec,
)

# Input Node
INPUT_SPEC = NodeSpec(
    type="input",
    label="Input",
    category="input",
    color="var(--color-teal)",
    icon="Download",
    description="Network input layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="shape",
            label="Input Shape",
            field_type="text",
            default="[1, 3, 224, 224]",
            description="Input tensor shape as JSON array (e.g., [1, 3, 224, 224])",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_input",
        engine="jinja2",
        content="""# Input shape: {{ config.shape }}
# This is a placeholder - actual input will come from DataLoader
""",
    ),
)

# Linear Node
LINEAR_SPEC = NodeSpec(
    type="linear",
    label="Linear",
    category="basic",
    color="var(--color-primary)",
    icon="Lightning",
    description="Fully connected layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="out_features",
            label="Output Features",
            field_type="number",
            required=True,
            min=1,
            description="Number of output features",
        ),
        ConfigFieldSpec(
            name="bias",
            label="Use Bias",
            field_type="boolean",
            default=True,
            description="Add learnable bias",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_linear",
        engine="jinja2",
        content="""nn.Linear({{ context.in_features }}, {{ config.out_features }}, bias={{ config.bias|lower }})""",
        default_context={"in_features": "None"},
    ),
)

# Conv2D Node
CONV2D_SPEC = NodeSpec(
    type="conv2d",
    label="Conv2D",
    category="basic",
    color="var(--color-purple)",
    icon="SquareHalf",
    description="2D convolutional layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="out_channels",
            label="Output Channels",
            field_type="number",
            required=True,
            min=1,
            description="Number of output channels",
        ),
        ConfigFieldSpec(
            name="kernel_size",
            label="Kernel Size",
            field_type="number",
            default=3,
            min=1,
            description="Size of convolving kernel",
        ),
        ConfigFieldSpec(
            name="stride",
            label="Stride",
            field_type="number",
            default=1,
            min=1,
            description="Stride of convolution",
        ),
        ConfigFieldSpec(
            name="padding",
            label="Padding",
            field_type="number",
            default=0,
            min=0,
            description="Zero-padding added to both sides",
        ),
        ConfigFieldSpec(
            name="dilation",
            label="Dilation",
            field_type="number",
            default=1,
            min=1,
            description="Spacing between kernel elements",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_conv2d",
        engine="jinja2",
        content="""nn.Conv2d({{ context.in_channels }}, {{ config.out_channels }}, kernel_size={{ config.kernel_size }}, stride={{ config.stride }}, padding={{ config.padding }}, dilation={{ config.dilation }})""",
        default_context={"in_channels": "None"},
    ),
)

# Flatten Node
FLATTEN_SPEC = NodeSpec(
    type="flatten",
    label="Flatten",
    category="basic",
    color="var(--color-primary)",
    icon="ListBullets",
    description="Flatten multi-dimensional input to 2D",
    framework=Framework.PYTORCH,
    config_schema=(),
    template=NodeTemplateSpec(
        name="pytorch_flatten",
        engine="jinja2",
        content="""nn.Flatten(start_dim=1)""",
    ),
)

# ReLU Node
RELU_SPEC = NodeSpec(
    type="relu",
    label="ReLU",
    category="basic",
    color="var(--color-accent)",
    icon="Lightning",
    description="Rectified Linear Unit activation",
    framework=Framework.PYTORCH,
    config_schema=(),
    template=NodeTemplateSpec(
        name="pytorch_relu",
        engine="jinja2",
        content="""nn.ReLU()""",
    ),
)

# Dropout Node
DROPOUT_SPEC = NodeSpec(
    type="dropout",
    label="Dropout",
    category="basic",
    color="var(--color-accent)",
    icon="Minus",
    description="Dropout regularization",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="p",
            label="Dropout Probability",
            field_type="number",
            default=0.5,
            min=0.0,
            max=1.0,
            description="Probability of an element to be zeroed",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_dropout",
        engine="jinja2",
        content="""nn.Dropout(p={{ config.p }})""",
    ),
)

# BatchNorm Node
BATCHNORM_SPEC = NodeSpec(
    type="batchnorm",
    label="Batch Normalization",
    category="basic",
    color="var(--color-accent)",
    icon="ChartLineUp",
    description="Batch normalization layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="eps",
            label="Epsilon",
            field_type="number",
            default=0.00001,
            min=0,
            description="Value added to denominator for numerical stability",
        ),
        ConfigFieldSpec(
            name="momentum",
            label="Momentum",
            field_type="number",
            default=0.1,
            min=0,
            max=1,
            description="Momentum for running mean and variance",
        ),
        ConfigFieldSpec(
            name="affine",
            label="Learnable Parameters",
            field_type="boolean",
            default=True,
            description="Enable learnable affine parameters",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_batchnorm",
        engine="jinja2",
        content="""nn.BatchNorm2d({{ context.num_features }}, eps={{ config.eps }}, momentum={{ config.momentum }}, affine={{ config.affine|lower }})""",
        default_context={"num_features": "None"},
    ),
)

# MaxPool2D Node
MAXPOOL_SPEC = NodeSpec(
    type="maxpool",
    label="MaxPool2D",
    category="basic",
    color="var(--color-purple)",
    icon="SquaresFour",
    description="2D max pooling layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="kernel_size",
            label="Kernel Size",
            field_type="number",
            default=2,
            min=1,
            description="Size of pooling window",
        ),
        ConfigFieldSpec(
            name="stride",
            label="Stride",
            field_type="number",
            default=2,
            min=1,
            description="Stride of pooling window",
        ),
        ConfigFieldSpec(
            name="padding",
            label="Padding",
            field_type="number",
            default=0,
            min=0,
            description="Zero-padding added to both sides",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_maxpool",
        engine="jinja2",
        content="""nn.MaxPool2d(kernel_size={{ config.kernel_size }}, stride={{ config.stride }}, padding={{ config.padding }})""",
    ),
)

# Softmax Node
SOFTMAX_SPEC = NodeSpec(
    type="softmax",
    label="Softmax",
    category="basic",
    color="var(--color-accent)",
    icon="Function",
    description="Softmax activation function",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="dim",
            label="Dimension",
            field_type="number",
            default=1,
            description="Dimension along which Softmax will be computed",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_softmax",
        engine="jinja2",
        content="""nn.Softmax(dim={{ config.dim }})""",
    ),
)

# Concat Node
CONCAT_SPEC = NodeSpec(
    type="concat",
    label="Concatenate",
    category="merge",
    color="var(--color-accent)",
    icon="GitMerge",
    description="Concatenate tensors along a dimension",
    framework=Framework.PYTORCH,
    allows_multiple_inputs=True,
    config_schema=(
        ConfigFieldSpec(
            name="dim",
            label="Dimension",
            field_type="number",
            default=1,
            description="Dimension along which to concatenate",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_concat",
        engine="jinja2",
        content="""# Concatenation happens in forward pass: torch.cat([x1, x2, ...], dim={{ config.dim }})""",
    ),
)

# Add Node
ADD_SPEC = NodeSpec(
    type="add",
    label="Add",
    category="merge",
    color="var(--color-accent)",
    icon="Plus",
    description="Element-wise addition",
    framework=Framework.PYTORCH,
    allows_multiple_inputs=True,
    config_schema=(),
    template=NodeTemplateSpec(
        name="pytorch_add",
        engine="jinja2",
        content="""# Element-wise addition happens in forward pass: x1 + x2""",
    ),
)

# Attention Node
ATTENTION_SPEC = NodeSpec(
    type="attention",
    label="Multi-Head Attention",
    category="advanced",
    color="var(--color-purple)",
    icon="Brain",
    description="Multi-head self-attention mechanism",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="embed_dim",
            label="Embedding Dimension",
            field_type="number",
            required=True,
            min=1,
            description="Total dimension of the model",
        ),
        ConfigFieldSpec(
            name="num_heads",
            label="Number of Heads",
            field_type="number",
            default=8,
            min=1,
            description="Number of attention heads",
        ),
        ConfigFieldSpec(
            name="dropout",
            label="Dropout",
            field_type="number",
            default=0.0,
            min=0.0,
            max=1.0,
            description="Dropout probability",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_attention",
        engine="jinja2",
        content="""nn.MultiheadAttention(embed_dim={{ config.embed_dim }}, num_heads={{ config.num_heads }}, dropout={{ config.dropout }}, batch_first=True)""",
    ),
)

# Custom Node
CUSTOM_SPEC = NodeSpec(
    type="custom",
    label="Custom Layer",
    category="advanced",
    color="var(--color-purple)",
    icon="Code",
    description="Custom user-defined layer",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="name",
            label="Layer Name",
            field_type="text",
            required=True,
            description="Name of the custom layer",
        ),
        ConfigFieldSpec(
            name="code",
            label="Python Code",
            field_type="text",
            required=True,
            description="Custom forward pass implementation",
        ),
        ConfigFieldSpec(
            name="output_shape",
            label="Output Shape",
            field_type="text",
            description="Expected output shape (optional)",
        ),
        ConfigFieldSpec(
            name="description",
            label="Description",
            field_type="text",
            description="Brief description of the layer functionality",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_custom",
        engine="jinja2",
        content="""# Custom Layer: {{ config.name }}
# {{ config.description }}
{{ config.code }}""",
    ),
)

# DataLoader Node
DATALOADER_SPEC = NodeSpec(
    type="dataloader",
    label="DataLoader",
    category="input",
    color="var(--color-teal)",
    icon="Database",
    description="Data loading and batching",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="batch_size",
            label="Batch Size",
            field_type="number",
            default=32,
            min=1,
            description="Number of samples per batch",
        ),
        ConfigFieldSpec(
            name="shuffle",
            label="Shuffle",
            field_type="boolean",
            default=True,
            description="Shuffle data at every epoch",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_dataloader",
        engine="jinja2",
        content="""DataLoader(dataset, batch_size={{ config.batch_size }}, shuffle={{ config.shuffle|lower }})""",
    ),
)

# Output Node
OUTPUT_SPEC = NodeSpec(
    type="output",
    label="Output",
    category="output",
    color="var(--color-green)",
    icon="Export",
    description="Define model output and predictions",
    framework=Framework.PYTORCH,
    config_schema=(),
    template=NodeTemplateSpec(
        name="pytorch_output",
        engine="jinja2",
        content="""# Output layer - no transformation needed""",
    ),
)

# Loss Node
LOSS_SPEC = NodeSpec(
    type="loss",
    label="Loss Function",
    category="output",
    color="var(--color-red)",
    icon="Target",
    description="Loss function for training",
    framework=Framework.PYTORCH,
    allows_multiple_inputs=True,
    config_schema=(
        ConfigFieldSpec(
            name="loss_type",
            label="Loss Type",
            field_type="select",
            default="cross_entropy",
            options=(
                ConfigOptionSpec(value="cross_entropy", label="Cross Entropy"),
                ConfigOptionSpec(value="mse", label="Mean Squared Error"),
                ConfigOptionSpec(value="mae", label="Mean Absolute Error"),
                ConfigOptionSpec(value="bce", label="Binary Cross Entropy"),
                ConfigOptionSpec(value="triplet", label="Triplet Loss"),
                ConfigOptionSpec(value="contrastive", label="Contrastive Loss"),
                ConfigOptionSpec(value="nll", label="Negative Log Likelihood"),
                ConfigOptionSpec(value="kl_div", label="KL Divergence"),
            ),
            description="Type of loss function",
        ),
    ),
    input_ports=(
        InputPortSpec(
            id="y_pred",
            label="Predictions",
            description="Model predictions (y_pred)",
        ),
        InputPortSpec(
            id="y_true",
            label="Ground Truth",
            description="True labels (y_true)",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_loss",
        engine="jinja2",
        content="""{% if config.loss_type == 'cross_entropy' %}nn.CrossEntropyLoss()
{%- elif config.loss_type == 'mse' %}nn.MSELoss()
{%- elif config.loss_type == 'mae' %}nn.L1Loss()
{%- elif config.loss_type == 'bce' %}nn.BCELoss()
{%- elif config.loss_type == 'triplet' %}nn.TripletMarginLoss()
{%- elif config.loss_type == 'contrastive' %}nn.CosineEmbeddingLoss()
{%- elif config.loss_type == 'nll' %}nn.NLLLoss()
{%- elif config.loss_type == 'kl_div' %}nn.KLDivLoss()
{%- else %}nn.CrossEntropyLoss()
{%- endif %}""",
    ),
    metadata={
        "input_ports_config": {
            "cross_entropy": ["y_pred", "y_true"],
            "mse": ["y_pred", "y_true"],
            "mae": ["y_pred", "y_true"],
            "bce": ["y_pred", "y_true"],
            "nll": ["y_pred", "y_true"],
            "kl_div": ["y_pred", "y_true"],
            "triplet": ["anchor", "positive", "negative"],
            "contrastive": ["input1", "input2", "label"],
        }
    },
)

# Empty Node
EMPTY_SPEC = NodeSpec(
    type="empty",
    label="Empty",
    category="utility",
    color="var(--color-gray)",
    icon="Circle",
    description="Placeholder for architecture planning",
    framework=Framework.PYTORCH,
    config_schema=(
        ConfigFieldSpec(
            name="note",
            label="Note",
            field_type="text",
            description="Notes about this placeholder",
        ),
    ),
    template=NodeTemplateSpec(
        name="pytorch_empty",
        engine="jinja2",
        content="""# Placeholder: {{ config.note }}""",
    ),
)

NODE_SPECS = (
    INPUT_SPEC,
    LINEAR_SPEC,
    CONV2D_SPEC,
    FLATTEN_SPEC,
    RELU_SPEC,
    DROPOUT_SPEC,
    BATCHNORM_SPEC,
    MAXPOOL_SPEC,
    SOFTMAX_SPEC,
    CONCAT_SPEC,
    ADD_SPEC,
    ATTENTION_SPEC,
    CUSTOM_SPEC,
    DATALOADER_SPEC,
    OUTPUT_SPEC,
    LOSS_SPEC,
    EMPTY_SPEC,
)
