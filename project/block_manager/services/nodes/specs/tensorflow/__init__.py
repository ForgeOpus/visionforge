from ..models import (
    ConfigFieldSpec,
    ConfigOptionSpec,
    Framework,
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
    description="Network input layer (NHWC format)",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="shape",
            label="Input Shape",
            field_type="text",
            default="[1, 224, 224, 3]",
            description="Input tensor shape as JSON array in NHWC format (e.g., [1, 224, 224, 3])",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_input",
        engine="jinja2",
        content="""# Input shape (NHWC): {{ config.shape }}
# This is a placeholder - actual input will come from dataset
""",
    ),
)

# Dense (Linear) Node
LINEAR_SPEC = NodeSpec(
    type="linear",
    label="Dense",
    category="basic",
    color="var(--color-primary)",
    icon="Lightning",
    description="Fully connected layer (Dense)",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="units",
            label="Units",
            field_type="number",
            required=True,
            min=1,
            description="Number of output units (neurons)",
        ),
        ConfigFieldSpec(
            name="activation",
            label="Activation",
            field_type="select",
            default="None",
            options=(
                ConfigOptionSpec(value="None", label="None"),
                ConfigOptionSpec(value="relu", label="ReLU"),
                ConfigOptionSpec(value="sigmoid", label="Sigmoid"),
                ConfigOptionSpec(value="tanh", label="Tanh"),
                ConfigOptionSpec(value="softmax", label="Softmax"),
            ),
            description="Activation function",
        ),
        ConfigFieldSpec(
            name="use_bias",
            label="Use Bias",
            field_type="boolean",
            default=True,
            description="Add learnable bias",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_dense",
        engine="jinja2",
        content="""layers.Dense({{ config.units }}, activation={% if config.activation == 'None' %}None{% else %}'{{ config.activation }}'{% endif %}, use_bias={{ config.use_bias|lower }})""",
    ),
)

# Conv2D Node
CONV2D_SPEC = NodeSpec(
    type="conv2d",
    label="Conv2D",
    category="basic",
    color="var(--color-purple)",
    icon="SquareHalf",
    description="2D convolutional layer (TensorFlow)",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="filters",
            label="Filters",
            field_type="number",
            required=True,
            min=1,
            description="Number of output filters (channels)",
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
            name="strides",
            label="Strides",
            field_type="number",
            default=1,
            min=1,
            description="Stride of convolution",
        ),
        ConfigFieldSpec(
            name="padding",
            label="Padding",
            field_type="select",
            default="valid",
            options=(
                ConfigOptionSpec(value="valid", label="Valid (no padding)"),
                ConfigOptionSpec(value="same", label="Same (preserve dimensions)"),
            ),
            description="Padding mode",
        ),
        ConfigFieldSpec(
            name="activation",
            label="Activation",
            field_type="select",
            default="None",
            options=(
                ConfigOptionSpec(value="None", label="None"),
                ConfigOptionSpec(value="relu", label="ReLU"),
                ConfigOptionSpec(value="sigmoid", label="Sigmoid"),
                ConfigOptionSpec(value="tanh", label="Tanh"),
            ),
            description="Activation function",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_conv2d",
        engine="jinja2",
        content="""layers.Conv2D({{ config.filters }}, {{ config.kernel_size }}, strides={{ config.strides }}, padding='{{ config.padding }}', activation={% if config.activation == 'None' %}None{% else %}'{{ config.activation }}'{% endif %})""",
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
    framework=Framework.TENSORFLOW,
    config_schema=(),
    template=NodeTemplateSpec(
        name="tensorflow_flatten",
        engine="jinja2",
        content="""layers.Flatten()""",
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
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="rate",
            label="Dropout Rate",
            field_type="number",
            default=0.5,
            min=0.0,
            max=1.0,
            description="Fraction of input units to drop",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_dropout",
        engine="jinja2",
        content="""layers.Dropout(rate={{ config.rate }})""",
    ),
)

# BatchNormalization Node
BATCHNORM_SPEC = NodeSpec(
    type="batchnorm",
    label="Batch Normalization",
    category="basic",
    color="var(--color-accent)",
    icon="ChartLineUp",
    description="Batch normalization layer",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="epsilon",
            label="Epsilon",
            field_type="number",
            default=0.001,
            min=0,
            description="Small constant for numerical stability",
        ),
        ConfigFieldSpec(
            name="momentum",
            label="Momentum",
            field_type="number",
            default=0.99,
            min=0,
            max=1,
            description="Momentum for moving average",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_batchnorm",
        engine="jinja2",
        content="""layers.BatchNormalization(epsilon={{ config.epsilon }}, momentum={{ config.momentum }})""",
    ),
)

# MaxPooling2D Node
MAXPOOL_SPEC = NodeSpec(
    type="maxpool",
    label="MaxPool2D",
    category="basic",
    color="var(--color-purple)",
    icon="SquaresFour",
    description="2D max pooling layer",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="pool_size",
            label="Pool Size",
            field_type="number",
            default=2,
            min=1,
            description="Size of pooling window",
        ),
        ConfigFieldSpec(
            name="strides",
            label="Strides",
            field_type="number",
            default=2,
            min=1,
            description="Stride of pooling window",
        ),
        ConfigFieldSpec(
            name="padding",
            label="Padding",
            field_type="select",
            default="valid",
            options=(
                ConfigOptionSpec(value="valid", label="Valid (no padding)"),
                ConfigOptionSpec(value="same", label="Same"),
            ),
            description="Padding mode",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_maxpool",
        engine="jinja2",
        content="""layers.MaxPooling2D(pool_size={{ config.pool_size }}, strides={{ config.strides }}, padding='{{ config.padding }}')""",
    ),
)

# ReLU Activation Node
RELU_SPEC = NodeSpec(
    type="relu",
    label="ReLU",
    category="basic",
    color="var(--color-accent)",
    icon="Lightning",
    description="Rectified Linear Unit activation",
    framework=Framework.TENSORFLOW,
    config_schema=(),
    template=NodeTemplateSpec(
        name="tensorflow_relu",
        engine="jinja2",
        content="""layers.ReLU()""",
    ),
)

# Softmax Activation Node
SOFTMAX_SPEC = NodeSpec(
    type="softmax",
    label="Softmax",
    category="basic",
    color="var(--color-accent)",
    icon="Function",
    description="Softmax activation function",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="axis",
            label="Axis",
            field_type="number",
            default=-1,
            description="Axis along which Softmax will be computed",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_softmax",
        engine="jinja2",
        content="""layers.Softmax(axis={{ config.axis }})""",
    ),
)

# Concat Node
CONCAT_SPEC = NodeSpec(
    type="concat",
    label="Concatenate",
    category="merge",
    color="var(--color-accent)",
    icon="GitMerge",
    description="Concatenate tensors along an axis",
    framework=Framework.TENSORFLOW,
    allows_multiple_inputs=True,
    config_schema=(
        ConfigFieldSpec(
            name="axis",
            label="Axis",
            field_type="number",
            default=-1,
            description="Axis along which to concatenate",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_concat",
        engine="jinja2",
        content="""# Concatenation happens in call method: layers.Concatenate(axis={{ config.axis }})([x1, x2, ...])""",
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
    framework=Framework.TENSORFLOW,
    allows_multiple_inputs=True,
    config_schema=(),
    template=NodeTemplateSpec(
        name="tensorflow_add",
        engine="jinja2",
        content="""# Element-wise addition happens in call method: layers.Add()([x1, x2])""",
    ),
)

# Multi-Head Attention Node
ATTENTION_SPEC = NodeSpec(
    type="attention",
    label="Multi-Head Attention",
    category="advanced",
    color="var(--color-purple)",
    icon="Brain",
    description="Multi-head self-attention mechanism",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="num_heads",
            label="Number of Heads",
            field_type="number",
            default=8,
            min=1,
            description="Number of attention heads",
        ),
        ConfigFieldSpec(
            name="key_dim",
            label="Key Dimension",
            field_type="number",
            required=True,
            min=1,
            description="Size of each attention head for query and key",
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
        name="tensorflow_attention",
        engine="jinja2",
        content="""layers.MultiHeadAttention(num_heads={{ config.num_heads }}, key_dim={{ config.key_dim }}, dropout={{ config.dropout }})""",
    ),
)

# DataLoader Node
DATALOADER_SPEC = NodeSpec(
    type="dataloader",
    label="DataLoader",
    category="input",
    color="var(--color-teal)",
    icon="Database",
    description="Data loading using PyDataset",
    framework=Framework.TENSORFLOW,
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
            description="Shuffle data",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_dataloader",
        engine="jinja2",
        content="""# PyDataset with batch_size={{ config.batch_size }}, shuffle={{ config.shuffle|lower }}""",
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
    framework=Framework.TENSORFLOW,
    config_schema=(),
    template=NodeTemplateSpec(
        name="tensorflow_output",
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
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="loss_type",
            label="Loss Type",
            field_type="select",
            default="categorical_crossentropy",
            options=(
                ConfigOptionSpec(value="categorical_crossentropy", label="Categorical Cross Entropy"),
                ConfigOptionSpec(value="sparse_categorical_crossentropy", label="Sparse Categorical Cross Entropy"),
                ConfigOptionSpec(value="mse", label="Mean Squared Error"),
                ConfigOptionSpec(value="mae", label="Mean Absolute Error"),
                ConfigOptionSpec(value="binary_crossentropy", label="Binary Cross Entropy"),
            ),
            description="Type of loss function",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_loss",
        engine="jinja2",
        content="""'{{ config.loss_type }}'  # Loss for model.compile()""",
    ),
)

# Empty Node
EMPTY_SPEC = NodeSpec(
    type="empty",
    label="Empty",
    category="utility",
    color="var(--color-gray)",
    icon="Circle",
    description="Placeholder for architecture planning",
    framework=Framework.TENSORFLOW,
    config_schema=(
        ConfigFieldSpec(
            name="note",
            label="Note",
            field_type="text",
            description="Notes about this placeholder",
        ),
    ),
    template=NodeTemplateSpec(
        name="tensorflow_empty",
        engine="jinja2",
        content="""# Placeholder: {{ config.note }}""",
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
    framework=Framework.TENSORFLOW,
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
            description="Custom call method implementation",
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
        name="tensorflow_custom",
        engine="jinja2",
        content="""# Custom Layer: {{ config.name }}
# {{ config.description }}
{{ config.code }}""",
    ),
)

NODE_SPECS = (
    INPUT_SPEC,
    LINEAR_SPEC,
    CONV2D_SPEC,
    FLATTEN_SPEC,
    RELU_SPEC,
    SOFTMAX_SPEC,
    DROPOUT_SPEC,
    BATCHNORM_SPEC,
    MAXPOOL_SPEC,
    CONCAT_SPEC,
    ADD_SPEC,
    ATTENTION_SPEC,
    DATALOADER_SPEC,
    OUTPUT_SPEC,
    LOSS_SPEC,
    EMPTY_SPEC,
    CUSTOM_SPEC,
)
