"�W
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1ffff��@Affff��@a��Y��?i��Y��?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1     =�@9     =�@A     =�@I     =�@a�"��µ?iޫE7�S�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1fffff�@9fffff�@Afffff�@Ifffff�@a���=�p�?i��?��?�Unknown
�HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1fffff��@9fffff��@Afffff��@Ifffff��@a_kk�z�?i)-�	Q�?�Unknown
tHost_FusedConv2D"sequential/activation/Relu(1������@9������@A������@I������@a�`��?i�-U�C�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffff��@9fffff��@Afffff��@Ifffff��@a����@
�?i(k����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1����̘�@9����̘�@A����̘�@I����̘�@a��g�?i�;JΠ�?�Unknown�
�	HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1fffffV�@9fffffV�@AfffffV�@IfffffV�@a��o�ਘ?i�Qf�?�Unknown
v
Host_FusedMatMul"sequential/activation_1/Relu(1�����%�@9�����%�@A�����%�@I�����%�@a����<�?i�{��?�Unknown
�HostReluGrad",gradient_tape/sequential/activation/ReluGrad(133333k~@933333k~@A33333k~@I33333k~@a�s�{�?i/Ԧ��P�?�Unknown
uHostMaxPool" sequential/max_pooling2d/MaxPool(1����̼|@9����̼|@A����̼|@I����̼|@a;��k#�?i`*V����?�Unknown
^HostGatherV2"GatherV2(133333�{@933333�{@A33333�{@I33333�{@a�;D��؉?iO;;w�#�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �j@9     �j@A     �j@I     �j@a�CI@��x?i�ͻ�U�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1������U@9������U@A������U@I������U@a&&��V9d?i�a�X+j�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     @I@9     @I@A     @I@I     @I@a����6�W?i;�s�u�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      G@9      G@A      G@I      G@a2�X�U?i�;#����?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(133333�D@933333�D@A33333�D@I33333�D@a��m-��S?i��9����?�Unknown
dHostDataset"Iterator::Model(133333sH@933333sH@Afffff&C@Ifffff&C@a9��E�Q?i��\����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffff�A@9fffff�A@Afffff�A@Ifffff�A@a����z�P?i�4����?�Unknown
�HostReluGrad".gradient_tape/sequential/activation_1/ReluGrad(1333333;@9333333;@A333333;@I333333;@a�U,˜wI?i���F��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1������:@9������:@A������:@I������:@ag	�^�I?i,� ͌��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�����?@9�����?@A     �9@I     �9@a��y�"�G?i�O�Մ��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1������4@9������4@A������4@I������4@a�f�ΥIC?i��S?W��?�Unknown
iHostWriteSummary"WriteSummary(1������.@9������.@A������.@I������.@a�Ś��<?iKMg��?�Unknown�
`HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a�m���<?iY�_�t��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff-@9ffffff-@Affffff-@Iffffff-@a�^�;?i8xC���?�Unknown
�HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      (@9      (@A      (@I      (@a��g�x6?ivwpش��?�Unknown
tHostSigmoid"sequential/activation_2/Sigmoid(1      &@9      &@A      &@I      &@a�r�I7�4?iĶY�G��?�Unknown
[HostAddV2"Adam/add(1ffffff%@9ffffff%@Affffff%@Iffffff%@a��`�f	4?i�.,���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@a��-qv�3?i���ZD��?�Unknown
e Host
LogicalAnd"
LogicalAnd(1      #@9      #@A      #@I      #@a�4�$�1?ih��}��?�Unknown�
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff"@9ffffff"@Affffff"@Iffffff"@a��azS:1?i>�����?�Unknown
�"HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������-@9������-@A      !@I      !@ask����/?i��/B���?�Unknown
}#HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1������ @9������ @A������ @I������ @aV�*%�/?ib�a����?�Unknown
x$HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����K@9�����K@A333333 @I333333 @a79^LV.?iF���x��?�Unknown
�%HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1333333@9333333@A333333@I333333@a�S+a6-?i��dL��?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@am�^.�v,?i�����?�Unknown
Y'HostPow"Adam/Pow(1      @9      @A      @I      @a���]7*?ivDtD���?�Unknown
V(HostMean"Mean(1������@9������@A������@I������@a���7}�)?i��G�T��?�Unknown
v)HostMul"%binary_crossentropy/logistic_loss/mul(1333333@9333333@A333333@I333333@a�U,˜w)?iep6���?�Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @aIp���W(?i��̵q��?�Unknown
�+HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1������@9������@A������@I������@a�#��'?i.i~7���?�Unknown
l,HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�#��'?i`0�p��?�Unknown
~-HostSelect"*binary_crossentropy/logistic_loss/Select_1(1������@9������@A������@I������@a���@Z8'?i��>���?�Unknown
�.HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a��g�x&?ih�j�K��?�Unknown
X/HostCast"Cast_2(1333333@9333333@A333333@I333333@aaX-�ظ%?i>��U���?�Unknown
�0HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1333333@9333333@A333333@I333333@aaX-�ظ%?it|���?�Unknown
t1HostReadVariableOp"Adam/Cast/ReadVariableOp(1������@9������@A������@I������@a��"�X%?i���rX��?�Unknown
V2HostAddN"AddN(1ffffff@9ffffff@Affffff@Iffffff@aC�`��$?i�z���?�Unknown
�3HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff@Affffff@Iffffff@aC�`��$?i�l�����?�Unknown
v4HostNeg"%binary_crossentropy/logistic_loss/Neg(1ffffff@9ffffff@Affffff@Iffffff@ax@a��#?i��N1)��?�Unknown
[5HostPow"
Adam/Pow_1(1      @9      @A      @I      @a���+չ"?i`���T��?�Unknown
~6HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@a�Z.S�!?iF��ot��?�Unknown
\7HostGreater"Greater(1������@9������@A������@I������@a����u?i�qp��?�Unknown
w8HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333@9333333@A333333@I333333@a�S+a6?i*��Y��?�Unknown
�9HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1333333@9333333@A333333@I333333@a�S+a6?i���C��?�Unknown
�:HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1ffffff@9ffffff@Affffff@Iffffff@am�^.�v?i{W8'��?�Unknown
];HostCast"Adam/Cast_1(1������@9������@A������@I������@aP!�U߶?i���?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@aP!�U߶?i��
����?�Unknown
z=HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1������@9������@A������@I������@a2��|�?iɖ�^���?�Unknown
|>HostSelect"(binary_crossentropy/logistic_loss/Select(1ffffff
@9ffffff
@Affffff
@Iffffff
@aؼ_�۷?i�)����?�Unknown
X?HostEqual"Equal(1������	@9������	@A������	@I������	@a�#��?i`���?��?�Unknown
�@HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1������	@9������	@A������	@I������	@a�#��?i�����?�Unknown
}AHostDivNoNan"'binary_crossentropy/weighted_loss/value(1������@9������@A������@I������@a���@Z8?i-�ab���?�Unknown
vBHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@aaX-�ظ?i�B&)g��?�Unknown
tCHostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a&&��V9?i9/����?�Unknown
vDHostExp"%binary_crossentropy/logistic_loss/Exp(1������@9������@A������@I������@a&&��V9?i������?�Unknown
oEHostReadVariableOp"Adam/ReadVariableOp(1������@9������@A������@I������@a���y?iBD�F��?�Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a���+չ?i��Y���?�Unknown
�GHostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1333333@9333333@A333333@I333333@a�Z.S�?ia;�*l��?�Unknown
�HHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a����!�?iA����?�Unknown
jIHostMean"binary_crossentropy/Mean(1�������?9�������?A�������?I�������?a2��|�
?iW���O��?�Unknown
vJHostSum"%binary_crossentropy/weighted_loss/Sum(1�������?9�������?A�������?I�������?a2��|�
?im�����?�Unknown
�KHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1�������?9�������?A�������?I�������?a2��|�
?i����'��?�Unknown
rLHostAdd"!binary_crossentropy/logistic_loss(1333333�?9333333�?A333333�?I333333�?a�U,˜w	?i4��v���?�Unknown
�MHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1333333�?9333333�?A333333�?I333333�?a�U,˜w	?i��kU���?�Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�#��?i2T�5S��?�Unknown
aOHostIdentity"Identity(1ffffff�?9ffffff�?Affffff�?Iffffff�?aC�`��?i�-7���?�Unknown�
�PHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aC�`��?i8�����?�Unknown
�QHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aC�`��?i����N��?�Unknown
vRHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a���y?i��Lɜ��?�Unknown
XSHostCast"Cast_3(1�������?9�������?A�������?I�������?a���y?i������?�Unknown
bTHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a���y?i��8��?�Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�Z.S�?i�fN~���?�Unknown
�VHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�(���z ?i#�h���?�Unknown
`WHostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a����!��>i��T���?�Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a����!��>im A:��?�Unknown
wYHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�#���>i)�V1j��?�Unknown
yZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aC�`���>i��#���?�Unknown
T[HostMul"Mul(1333333�?9333333�?A333333�?I333333�?a�Z.S��>iG�����?�Unknown
�\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1333333�?9333333�?A333333�?I333333�?a�Z.S��>i�Y����?�Unknown
�]HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1333333�?9333333�?A333333�?I333333�?a�Z.S��>i      �?�Unknown*�V
�HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1     =�@9     =�@A     =�@I     =�@a	��/�?i	��/�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1fffff�@9fffff�@Afffff�@Ifffff�@aI�r���?i1�N���?�Unknown
�HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1fffff��@9fffff��@Afffff��@Ifffff��@a'�����?ib$�?�Unknown
tHost_FusedConv2D"sequential/activation/Relu(1������@9������@A������@I������@a�{QLH$�?i�5���?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffff��@9fffff��@Afffff��@Ifffff��@a'Q��ի?i�j�8���?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1����̘�@9����̘�@A����̘�@I����̘�@a��5���?i���;��?�Unknown�
�HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1fffffV�@9fffffV�@AfffffV�@IfffffV�@a_���i�?i�t0z{7�?�Unknown
vHost_FusedMatMul"sequential/activation_1/Relu(1�����%�@9�����%�@A�����%�@I�����%�@a���WΠ?i��1�`D�?�Unknown
�	HostReluGrad",gradient_tape/sequential/activation/ReluGrad(133333k~@933333k~@A33333k~@I33333k~@a�&�a��?i��A��A�?�Unknown
u
HostMaxPool" sequential/max_pooling2d/MaxPool(1����̼|@9����̼|@A����̼|@I����̼|@a����?i���0�?�Unknown
^HostGatherV2"GatherV2(133333�{@933333�{@A33333�{@I33333�{@a������?i1�C��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �j@9     �j@A     �j@I     �j@a����Ƌ?i}s����?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1������U@9������U@A������U@I������U@a�=�T{v?i�����?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     @I@9     @I@A     @I@I     @I@a�v��Gj?i�d*��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      G@9      G@A      G@I      G@a��Ŕ\�g?i�T����?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(133333�D@933333�D@A33333�D@I33333�D@aU��Z#�e?iwR�����?�Unknown
dHostDataset"Iterator::Model(133333sH@933333sH@Afffff&C@Ifffff&C@a��M���c?iP��j��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffff�A@9fffff�A@Afffff�A@Ifffff�A@aY���v�b?it��x!�?�Unknown
�HostReluGrad".gradient_tape/sequential/activation_1/ReluGrad(1333333;@9333333;@A333333;@I333333;@a��_*tO\?i
�ϛ�/�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1������:@9������:@A������:@I������:@a4�8���[?ir���=�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�����?@9�����?@A     �9@I     �9@a�߹�|�Z?ib�J�J�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1������4@9������4@A������4@I������4@a�m�c�pU?i�ƺ�U�?�Unknown
iHostWriteSummary"WriteSummary(1������.@9������.@A������.@I������.@aa��yVP?i�zf�]�?�Unknown�
`HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a�Cq�9O?iBW��be�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff-@9ffffff-@Affffff-@Iffffff-@aϭ6��N?i�$B0	m�?�Unknown
�HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      (@9      (@A      (@I      (@aQi'���H?i�n<�Gs�?�Unknown
tHostSigmoid"sequential/activation_2/Sigmoid(1      &@9      &@A      &@I      &@a� d@��F?iχ�^y�?�Unknown
[HostAddV2"Adam/add(1ffffff%@9ffffff%@Affffff%@Iffffff%@a��)t
FF?i2�)�~�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@an0�F?i��5��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      #@9      #@A      #@I      #@a`3?C��C?i�g���?�Unknown�
vHostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff"@9ffffff"@Affffff"@Iffffff"@a��w�&C?i�($aҍ�?�Unknown
� HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������-@9������-@A      !@I      !@a��{���A?i��J�>��?�Unknown
}!HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1������ @9������ @A������ @I������ @a�TGA?i�\O����?�Unknown
x"HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����K@9�����K@A333333 @I333333 @a}�-��@?iL�1�ǚ�?�Unknown
�#HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1333333@9333333@A333333@I333333@a�7�<@?iea�֞�?�Unknown
�$HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a3R�k�??i$�n[ˢ�?�Unknown
Y%HostPow"Adam/Pow(1      @9      @A      @I      @a���:�$=?i�-�o��?�Unknown
V&HostMean"Mean(1������@9������@A������@I������@aQ솲�<?i�~,0��?�Unknown
v'HostMul"%binary_crossentropy/logistic_loss/mul(1333333@9333333@A333333@I333333@a��_*tO<?i�ʱ���?�Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a�ꑶ;?i���?�Unknown
�)HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1������@9������@A������@I������@a���	"�:?i�@ŹG��?�Unknown
l*HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a���	"�:?i�x^���?�Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1������@9������@A������@I������@am�u���9?i��%]ֺ�?�Unknown
�,HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aQi'���8?i��"����?�Unknown
X-HostCast"Cast_2(1333333@9333333@A333333@I333333@a4L�ئ%8?i���k���?�Unknown
�.HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1333333@9333333@A333333@I333333@a4L�ئ%8?i�� ���?�Unknown
t/HostReadVariableOp"Adam/Cast/ReadVariableOp(1������@9������@A������@I������@a�=�P�7?i5#����?�Unknown
V0HostAddN"AddN(1ffffff@9ffffff@Affffff@Iffffff@a/��}P7?i�*ܒ���?�Unknown
�1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff@Affffff@Iffffff@a/��}P7?i<�����?�Unknown
v2HostNeg"%binary_crossentropy/logistic_loss/Neg(1ffffff@9ffffff@Affffff@Iffffff@aQ���;5?i�4yr��?�Unknown
[3HostPow"
Adam/Pow_1(1      @9      @A      @I      @a�נ��4?i)�5��?�Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@a��R���3?ip�����?�Unknown
\5HostGreater"Greater(1������@9������@A������@I������@aRchV^|1?i|��<���?�Unknown
w6HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333@9333333@A333333@I333333@a�7�<0?i������?�Unknown
�7HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1333333@9333333@A333333@I333333@a�7�<0?iJ]�d���?�Unknown
�8HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1ffffff@9ffffff@Affffff@Iffffff@a3R�k�/?i�~����?�Unknown
]9HostCast"Adam/Cast_1(1������@9������@A������@I������@a5J[��.?ir�s����?�Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@a5J[��.?i�i����?�Unknown
z;HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1������@9������@A������@I������@a��J��-?i�/� ~��?�Unknown
|<HostSelect"(binary_crossentropy/logistic_loss/Select(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��Kz+?i���5��?�Unknown
X=HostEqual"Equal(1������	@9������	@A������	@I������	@a���	"�*?i,m����?�Unknown
�>HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1������	@9������	@A������	@I������	@a���	"�*?if	�i���?�Unknown
}?HostDivNoNan"'binary_crossentropy/weighted_loss/value(1������@9������@A������@I������@am�u���)?i��Pi'��?�Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@a4L�ئ%(?iS.�é��?�Unknown
tAHostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a�=�T{&?i$�	y��?�Unknown
vBHostExp"%binary_crossentropy/logistic_loss/Exp(1������@9������@A������@I������@a�=�T{&?i�5U.y��?�Unknown
oCHostReadVariableOp"Adam/ReadVariableOp(1������@9������@A������@I������@a���+�%?i�����?�Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a�נ��$?i�.9� ��?�Unknown
�EHostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1333333@9333333@A333333@I333333@a��R���#?i��^`��?�Unknown
�FHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a6FF5� ?i�&�j��?�Unknown
jGHostMean"binary_crossentropy/Mean(1�������?9�������?A�������?I�������?a��J��?i�]X�Z��?�Unknown
vHHostSum"%binary_crossentropy/weighted_loss/Sum(1�������?9�������?A�������?I�������?a��J��?i���nJ��?�Unknown
�IHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1�������?9�������?A�������?I�������?a��J��?id�<:��?�Unknown
rJHostAdd"!binary_crossentropy/logistic_loss(1333333�?9333333�?A333333�?I333333�?a��_*tO?ic`^���?�Unknown
�KHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1333333�?9333333�?A333333�?I333333�?a��_*tO?ib��3���?�Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a���	"�?i]���?�Unknown
aMHostIdentity"Identity(1ffffff�?9ffffff�?Affffff�?Iffffff�?a/��}P?i�E�����?�Unknown�
�NHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a/��}P?i1��dI��?�Unknown
�OHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a/��}P?i������?�Unknown
vPHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a���+�?i8���?�Unknown
XQHostCast"Cast_3(1�������?9�������?A�������?I�������?a���+�?izM�K^��?�Unknown
bRHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a���+�?i��|��?�Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��R���?i�Ǿ[���?�Unknown
�THostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?ao��f�Q?i<���=��?�Unknown
`UHostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a6FF5�?i.�!���?�Unknown
uVHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a6FF5�?i�^N[H��?�Unknown
wWHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a���	"�
?i�����?�Unknown
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a/��}P?i��1��?�Unknown
TYHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a��R���?ig�3!`��?�Unknown
�ZHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1333333�?9333333�?A333333�?I333333�?a��R���?i�����?�Unknown
�[HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1333333�?9333333�?A333333�?I333333�?a��R���?i�������?�Unknown