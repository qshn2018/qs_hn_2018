《利用Python进行数据分析》
=================
numpy
-------------------
### 导入
``` python
import numpy as np                                         # 行业习惯
```
### 数据类型（ndarray）
#### 创建数组
##### 创建一维数组
``` python
a1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])                 # 创建一维数组

b1 = arange(1, 16, 2)        # 创建以1开头，16结尾，不包括16，步长为2的一维数组

c1 = np.zeros(10)          # 创建一维零数组
d1 = np.ones(10)           # 创建一维一数组
e1 = np.empty(10)          # 创建一维数组，只分配内存空间，不填充任何数值
```
##### 创建二维数组
``` python
a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])           # 创建二维数组

b2 = np.arange(1, 16, 2).reshape((4,2))                    # 将一维b1重新组成4行2列

c2 = np.zeros((3, 4))      # 创建二维零数组
d2 = np.ones((3, 4))       # 创建二维数组
e2 = np.empty((3, 4))      # 创建二维矩阵

f1 = np.eye(3)             # 创建二维单位矩阵
f2 = np.identity(3)        # 同上
```
##### 其他创建函数
``` python
c3 = np.zeros_like(c2)     # 创建和c2相同维度的数组
d3 = np.ones_like(d2)      # 创建和d2相同维度的数组
e3 = np.empty_like(f2)     # 创建和f2相同维度的数组
```
#### 数组属性访问
``` python
a1.ndim                    # 访问属性：维度(轴的个数)
a1.shape                   # 访问属性：轴长度（第一维轴长度，第二维轴长度，……）
a1.size                    # 访问属性：访问数组长度，即元素个数
a1.dtype                   # 访问属性：元素数据格式
```
#### 数据格式
##### 数据格式更改
``` python
a1 = np.array([1, 2, 3], dtype = np.float64)               # 创建时指定数据格式

a1.astype(np.int64)        # 显式地转换dtype，调用astype无论如何都会创建出一个新的数组，就算dtype相同也会
#将浮点数转换成整数，小数部分会被截断
```
##### 数据格式一览
Ｎumpy数据类型         |            类型代码        |         说明
---------------------| ----------------------------|--------------------------
int8   uint8 | i1  u1 | 有符号和无符号的8位（1个字节）整型
int16  uint16 | i2  u2 | 有符号和无符号的16位（2个字节）整型
int32  uint32 | i4  u4 | 有符号和无符号的32位（4个字节）整型
int64  uint64 | i8  u8 | 有符号和无符号的64位（8个字节）整型
float16 | f2 | 半精度浮点数
float32 | f4或f | 标准的单精度浮点数。与Ｃ的float兼容
float64 | f8或d | 标准的双精度浮点数。与Ｃ的double和Ｐython的float对象兼容
float128 | f16或g | 扩展精度浮点数
complex64 complex128 | c8 c16 | 分别用两个32位、64位或128位浮点数表示的复数
bool | ? | 存储Ｔrue和Ｆalse值的布尔类型
object | Ｏ | Ｐython对象类型
string_ | Ｓ | 固定长度的字符串类型（每个字符1个字节），例如要创建一个长度为10的字符串，应使用Ｓ10
unicode_ | Ｕ | 固定长度的unicode类型（字节数由平台决定）。跟字符串的定义方式一样（Ｕ10）


#### 索引和切片
##### 一维数组索引和切片
``` python
a1[0]                      # 一维数组索引，从0开始
a1[0:2]                    # 一维切片，前闭后开
```
##### 多维数组索引和切片
``` python
a2[0][0]                   # 多维递归索引
a2[0, 0]                   # 多维直接访问

# 返回相同维度的数组视图
s1 = a2[0:2][0:2]          # 多维递归切片
s1 = a2[0:2, 0:2]          # 多维直接切片
s1                         # --> array([[1, 2], 
                                        [4, 5]])
a2.ndim                    # --> 2                                        
s1.ndim                    # --> 2

s2 = a2[2:, :]             # 前后都是切片
s2                         # --> array([[7, 8, 9]])，注意有两个中括号
s2.ndim                    # --> 2，还是二维的
# 返回低一个维度的数组
s3 = a2[2]                 # 多维数组，省略索引
s3.ndim                    # --> 1
s4 = a2[2, :]              # 前后只有一个逗号
s4.ndim                    # --> 1
```
##### 切片是视图
``` python
# 数组切片是原始数据的视图，
# 这意味着，数据不会被复制，视图上的任何修改都会直接反映到原始数据上
as = a1[3:6]               # --> 4,5,6（视图）
as[0] = 0                  # --> 0,5,6（视图）
a                          # --> 1,2,3,0,5,6,7,8,9（原始数据--改变）

# 如果你想得到切片的一个副本而不是视图，需要显式地用copy()方法进行复制
a = np.array([1,2,3,4,5,6,7,8,9])
a_s = a[3:6].copy()        # --> 4,5,6（副本）
a_s[0] = 0                 # --> 0,5,6（副本）
a                          # --> 1,2,3,4,5,6,7,8,9（原始数据--没变）
```
##### bool索引 ------ (副本)
对数组作比较运算，会返回一个相同维度的bool数组，符合条件的返回Ｔrue，否则返回Ｆalse
``` python
a2[a2>5]                   # --> array([6, 7, 8, 9])

row = [1, 0, 1]
a2[row == 1]               # --> array([1, 2, 3], 
                                       [7, 8, 9])
a2[-(row == 1)]            # 要选择不等于除了!=（不等于号）还可以用-（负号）
```
bool索引可以和索引切片一起使用
``` python
a2[row == 1, 1]            # --> array([2, 8])
a2[row == 1, 1:]           # --> array([2, 3], 
                                       [8, 9])
```
选择多个条件，需要逻辑运算
``` python
a2[(row == 1) | (row == 0)]# -->array([[1, 2, 3], 
                                       [4, 5, 6], 
                                       [7, 8, 9]])
# Python关键字and和or在bool数组中无效
```

##### 花式索引 ------ (副本)
利用整数数组以特定顺序进行索引
``` python
a2[[2, 0, 1]]              # --> array([[7, 8, 9], 
                                        [1, 2, 3], 
                                        [4, 5, 6]])
a2[[-1, -2, -3]]           # --> array([[7, 8, 9], 
                                        [4, 5, 6], 
                                        [1, 2, 3]])
```
传入多个索引数组，会返回一个一维数组，对应索引
``` python
a2[[1, 2], [2, 0]]         # --> array([6, 7])
# 最终选取的是（1, 2）（2, 0）两个元素
```
选取矩阵子列
``` python
# 方法一：先按特定顺序排序选出行，再索引全部行，再按特定顺序选取列
a[[1, 2]],[:, [2, 0]]      # --> array([[6, 4], 
                                        [9, 7]])

# 方法二：使用np.ix_函数，它可以将两个一维数组转换成一个用于选取方形区域的索引器
a[np.ix_([1, 2], [2, 0])]  # --> array([[6, 4], 
                                        [9, 7]])
```

#### 运算
##### 算数运算
维度相等的数组之间的任何算数运算，都会将运算应用到元素级  
加减乘除乘方 + - * / **  
数组和标量之间的算术运算也会将那个标量值”传播“到各个元素  
加减乘除乘方 + - * / **  

##### 比较运算
返回bool数组：> < >= <= == !=

##### 线性代数
矩阵乘法（点积）
``` python
x.dot(y)                   # 既是方法
np.dot(x, y)               # 也是函数
```
转置 ------ (视图)
``` python
a2.T
a2.transpose()
swapaxes((1,2))
```
|线性代数函数|说明|
|---|---|
|diag|以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换成方阵（非对角线元素为0）|
|dot|矩阵乘法|
|trace|计算对角线元素的和|
|det|计算矩阵行列式|
|eig|计算方阵的特征值和特征向量|
|inv|计算方阵的逆|
|pinv|计算矩阵的Moore-Penrose伪逆|
|qr|计算QR分解|
|svd|计算奇异值分解（SVD）|
|solve|解线性方程组Ax = b，其中A为一个方阵|
|lstsq|计算Ax = b的最小二乘解|

### 通用函数（ufunc）
通用函数，是一种对ndarray中的数据进行“元素级”运算的函数  

|一元函数 | 说明|
|--- | -----|  
|abs、fabs | 计算整数、浮点数或复数的绝对值。对于非复数值，可以使用更快的fabs|
|sqrt      | 计算各元素的平方根。相当于a ** 0.5|
|square    | 计算各元素的平方。相当于a ** 2|
|exp       | 计算各元素的指数e ^ x|
|log、log10、log2、log1p | 分别为自然对数（底数为e）、底数为10的log、底数为2的log、log(1 + x)|
|sign      | 计算各元素的正负号：1（正数）、0（零）、-1（负数）|
|ceil      | 计算各元素的ceiling值，即大于等于该值的最小整数|
|floor     | 计算各元素的floor值，即小于等于该值的最大整数|
|rint      | 将各元素值四舍五入到最接近的整数，保留dtype|
|modf      | 将数组的小数和整数部分以两个独立数组的形式返回|
|isnan     | 返回一个表示“哪些值是NaN（not a number）”的bool数组|
|isfinite、isinf | 分别返回一个表示“哪些元素是无穷的（非inf，非NaN）”或“哪些元素是无穷的”的bool数组|
|cos、cosh、sin、sinh、tan、tanh | 普通型和双曲线型三角函数|
|arccos、arccosh、arcsin、arcsinh、arctan、arctanh | 反三角函数|
|logical_not | 计算各元素的notx的真值。相当于-arr|

|二元函数|说明|
|---|---|
|add|将数组中对应的元素相加|
|subtract|从第一个数组中减去第二个数组中的元素|
|multiply|数组元素相乘|
|divide、floor_divide|除法或向下圆整除法（丢弃余数）|
|power|对第一个数组中的元素A，根据第二个数组中的相应元素B，计算A ^ B|
|maximum、fmax|元素级的最大值计算。fmax将忽略NaN|
|minimum、fmin|元素级的最小值计算。fmin将忽略NaN|
|mod|元素级的求模计算（除法的余数）|
|copysign|将第二个数组中的值的符号复制给第一个数组中的值|
|greater、greater_equal、less、less_equal、equal、not_equal|执行元素级的比较运算，最终产生bool数组。相当于中缀运算符>、>=、<、<=、==、!=|
|logical_and、logical_or、logicalxor|执行元素级的真值逻辑运算。相当于中缀运算符&、|、^|

### 数据处理
#### 将条件逻辑表述为数组运算 ------ /类似于：条件索引赋值/
`np.where()`函数，第一个参数为bool数组，如果为True变为第二个参数，如果为False变为第三个参数
第二第三个参数可以为数组或标量，数组的话大小可以不相等
``` python
np.where(a2 > 3, 0, -1)    # --> array([[-1, -1, -1], 
                                        [0, 0, 0], 
                                        [0, 0, 0]])
```
更复杂的逻辑：
假设有两个bool类型数组cond1和cond2
``` python
np.where(cond1 & cond2), 0,                                # 如果cond1和cond2都为True，则为0
            np.where(cond1, 1,                             # 如果只有cond1为True，则为1
               np.where(cond2, 2, 3)))                     # 如果只有cond2为True，则为2, 都为False,则为3

# 可以利用“bool值在计算过程中可以被当作0或1处理”将上面的写成下面的式子
1 * （cond1 & -cond2） + 2 * （cond2 & -cond1） + 3 * -（cond1 | cond2）
```

#### 数学和统计方法
求和
``` python
a2.sum()                   # 全体元素求和
a2.sum(0)                  # 0--对列求和
a2.sum(axis = 1)           # axis参数，1--对行求和

np.sum(a2)                 # sum函数既是方法，也是numpy的函数
np.sum(a2, 0)
np.sum(a2, axis = 1)
```
求最值
``` python
a2.max()                   # 全体元素最大值
a2.max(0)                  # 0--列
a2.max(axis = 1)           # 1--行

np.min(a2)                 # 全体元素最小值
np.min(a2, 0)              # 0--列
np.min(a2, axis = 1)       # 1--行

a2.argmax()                # 全体元素最大值的个数索引
a2.argmax(0)               # 每列最大值的索引
a2.argmax(axis = 1)        # 每行最大值的索引

np.argmin(a2)
np.argmin(a2, 0)
np.argmin(a2, axis = 1)
```
平均值
``` python
a.mean()                   # 全体元素平均值
a.mean(0)                  # 0--列，1--行
np.mean(a2, axis = 1)      # axis名可缺省
```
方差
``` python
a.var()                    # 全体元素求方差
a.var(0)                   # 0--列，1--行
np.var(a2, axis = a)       # axis名可缺省
```
标准差
``` pthon
a.std()                    # 全体元素求标准差
a.std(0)                   # 0--列，1--行
np.std(a2, axis = 1)       # axis名可缺省
```
中值
``` pthon
a.median()                 # 全体元素求中值
a.median(0)                # 0--列，1--行
np.median(a2, axis = 1)    # axis名可缺省
```
累计和&累计积
``` python
a.cumsum()
a.cumsum(0)
np.cumsum(a2, 1)

a.cumprod()
a.cumprod(0)
np.cumprod(a2, axis = 1)
```
|方法|说明|
|---|---|
|sum|对数组中全部或某轴向的元素求和。零长度的数组的sum为0|
|mean|算术平均值。零长度的数组的mean为NaN|
|std、var|分别为标准差和方差，自由度可调|
|min、max|最大值和最小值|
|argmin、argmax|分别为最大和最小元素的索引|
|cumsum|所有元素的累计和|
|cumprod|所有元素的累计积|

#### 用于bool类型数组的方法
bool数组用上面的数学统计方法会被强制转换成1和0。因此sum经常被用来对bool类型数组中的True计数
``` python
（a2 > 3).sum()
```
另外还有两个方法：any和all
any用于测试数组中是否存在True
all用于检查数组是否全是True
这两个方法也能用于非bool数组，所有非0元素都会被当成True
``` python
b = np.array([True, False, True, False])
b.any()                    # --> True
b.all()                    # --> False
```

#### 排序
计算数组分位数最简单的方法是对其进行排序，然后选取特定位置的值
``` python
a1.sort()                  # 从小到大排序，轴向默认为行
a2.sort(0)                 # 二维需要提供轴向，0--列，1--行
a2.sort(1)                 # sort方法是改变原始数据

np.sort(a2)                # 默认对行排列
np.sort(a2, 0)             # 0--列，1--行
np.sort(a2, 1)             # sort()函数是创建副本
```
#### 唯一化以及其他的集合逻辑
`np.unique()`函数用于找出数组中的唯一值并返回已排序的结果
``` python
np.unique([4, 4, 1, 2, 2, 3])                              # --> array([1, 2, 3, 4])
# 等价于
sorted(set(4, 4, 1, 2, 2, 3))
```
`np.in1d()`函数用于测试数组中的值在另一个数组中的是否存在
``` python
np.in1d([6, 0, 3, 2, 5, 6], [2, 3, 6])                  # --> array([True, False, True, True, False, False], dtype = bool)
```
|集合函数|说明|
|---|---|
|unique(x)|计算x中的唯一元素，并返回有序结果|
|intersect1d(x, y)|计算x和y中的公共元素，并返回有序结果|
|union1d(x, y)|计算x和y的合集，并返回有序结果|
|in1d(x, y)|得到一个表示“x的元素是否包含于y”的bool数组|
|setdiff1d(x, y)|集合的差，即元素在x中且不在y中|
|setxor1d(x, y)|集合的对称差（异或），即存在于一个数组中但不同时存在于两个数组中的元素|

### 用于数组的文件输入输出
numpy能够读写磁盘上的文本数据或二进制数据
#### 将数组以二进制格式保存到磁盘
`np.save()`和`np.load()`是读写磁盘数组数据的两个主要函数。
默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为`.npy`的文件中的。
如果文件路径末尾没有扩展名`.npy`，则该扩展名会被自动加上。
``` python
np.save('myfile', a2)                                     # 文件名，要存的数据
```
可以通过`np.load()`读取磁盘上的数组
``` python
np.load('myfile.npy')                                     # 文件名
```
通过`np.savez()`可以将多个数组保存在一个压缩文件中，将数组以关键字参数的形式传入。
加载`.npz`文件时，会得到一个类似字典的对象，该对象会对各个数组进行延迟加载：
``` python
np.savez('zipfile', a = a1, b = a2)

z = np.load('zipfile.npz')     
z['a']                                                     # --> array([1, 2, 3, 4, 5, ,6 ,7 ,8 ,9])
```

#### 存取文本文件
``` python
np.loadtxt('myfile', delimiter = ',')                      # 加载文本文件，分隔符为逗号“，”
# np.genfromtxt跟loadtxt差不多，只不过是面向的是结构化数组和缺失数据处理

# np.savetxt执行的是相反的操作：将数组写到以某种分隔符隔开的文本文件中

```
### 随机数生成
np.random模块对Python内置的random进行了补充，增加了一些用于高效生成多种概率分布的样本值的函数。

|函数|说明|
|---|---|
|seed|确定随机数生成器的种子|
|permutation|返回一个序列的随机排列或返回一个随即排列的范围|
|shuffle|对一个序列就地随即排列|
|rand|产生均匀分布的样本值|
|randint|从给定的上下限范围内随机选取整数|
|randn|产生正态分布（平均数为0,标准差为1）的样本值，类似于MATLAB接口|
|binomial|产生二项分布的样本值|
|normal|产生正态（高斯）分布的样本值|
|beta|产生Beta分布的样本值|
|chisquare|产生卡方分布的样本值|
|gamma|产生Gamma分布的样本值|
|uniform|产生在0到1中（前闭后开）均匀分布的样本值|


### 范例：随机漫步
纯Python
``` python
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += [step]
    walk.append(position)
```
numpy
``` python
nsteps = 1000
draws = np.random.randint(0, 2, size = nsteps)
steps = np.where(draws > 0, 1, -1)
walk = step.cumsum()

walk.min()
walk.max()

(np.abs(walk) >= 10).argmax()
```

#### 一次模拟多个随机漫步

``` python
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size = (nwalks, nsteps))   # 5000行1000列随机矩阵
steps = np.where(draws > 0, 1, -1)                         # 变成1和-1
walks = steps.cumsum(1)                                    # 对行计算累计和
walks

walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30.sum()                                               # 到达30或-30的数量
times = (np.abs(walks[hits30]) >= 30).argmax(1)
times.mean()
```




numpy高级应用
------------------




pandas
-------------------
### 0-导入
``` python
from pandas import Series, DataFrame                       # 因为Series和DataFrame用的次数非常多，所以将其引入本地命名空间中会更方便
import pandas                                              # 行业约定
```

### 1-数据结构
#### Series
##### 创建
``` python
obj1 = Series([4, 7, -5, 3])                               # 创建Ｓeries对象
obj1                                                       # --> 0     4
                                                                 1     7
                                                                 2    -5
                                                                 3     3
```
##### 可以将Ｓeries看成一个定长的有序字典
如果数据存放在一个Ｐython字典中，也可以直接通过这个字典创建Ｓeries
``` python
sdata = ['ohio':35000, 'texas':71000, 'oregon':16000, 'utah':5000]
obj3 = Series(sdata)
obj3                                                       # --> ohio      35000
                                                                 oregon    16000
                                                                 texas     71000
                                                                 utah       5000
```
如果只传入一个字典，则结果Ｓeries中的索引就是原字典的键（有序排列）
``` python
states = ['california', 'ohio', 'oregon', 'texas']
obj4 = Series([sdata, index = states])
obj4                                                       # --> california    NaN
                                                                 ohio        35000
                                                                 oregon      16000
                                                                 texas       71000
# 如果找不到索引对应的项，结果就是ＮaＮ（not a number）
```
可以用在许多原本需要字典参数的函数中
``` python
'b' in obj2                                                # --> True
'e' in obj2                                                # --> Flase
```
##### 访问属性：值、索引（index）、name属性
``` python
obj1.values                                                # --> array([4, 7, -5, 3])
obj1.index                                                 # --> RangeIndex(start = 0, stop = 4, step = 1)

obj1.index = ['bob', 'steve', 'jeff', 'ryan']              # Series的索引可以通过赋值的方式就地修改
obj1                                                       # --> bob      4
                                                                 steve    7
                                                                 jeff    -5
                                                                 ryan     3

# Ｓeries对象本身及其索引都有一个name属性，该属性跟pandas其他的关键功能关系非常密切
obj4.name = 'population'                                   # 给Ｓeries对象的name属性赋值
obk4.index.name = 'states'                                 # 给Ｓeries对象的index的name属性赋值
obj4                                                       # --> states
                                                                 california    NaN
                                                                 ohio        35000
                                                                 oregon      16000
                                                                 texas       71000
```
##### 创建时自定义索引
``` python
obj2 = Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c']) # 创建时自定义索引
obj2                       # --> d     4
                                 b     7
                                 a    -5
                                 c     3
obj2.index                 # --> Index(['d', 'b', 'a', 'c'], dtype = 'object')
```

##### 索引
``` python
obj2['a']                  # --> -5
obj2['d'] = 6              # 通过赋值更改数据

# 通过一个list索引多个数据
obj2[['c', 'a', 'd']]      # --> c     3
                                 a    -5
                                 d     6
```
numpy数组运算都会保留索引和值之间的链接
```python
# 通过bool数组进行过滤
obj2[obj > 0]              # --> d    6
                                 b    7
                                 c    3

# 标量乘法
obj2 * 2                   # --> d     12
                                 b     14
                                 a    -10
                                 c     6
# 应用数学函数
np.exp(obj2)               # --> d     403.428793
                                 b    1096.633158
                                 a       0.006738
                                 c      20.085537
```

##### 数据缺失
pandas的`isnull()`和`notnull()`函数可用于检测缺失数据
``` python
pd.isnull(obj4)            # --> california    True
                                 ohio         False
                                 oregon       False
                                 texas        False
pd.notnull(obj4)           # --> california    False
                                 ohio           True
                                 oregon         True
                                 texas          True

# Series也有类似的实例方法
obj4.isnull()              # --> california    True
                                 ohio         False
                                 oregon       False
                                 texas        False
```
##### 数据对齐
Series最重要的一个功能是：在算术运算中，会自动对齐不同索引的数据
``` python
obj3 + obj4                # --> california    NaN
                                 ohio        70000
                                 oregon      32000
                                 texas      142000
                                 utah          NaN
```
#### DataFrame
ＤataＦrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型。
ＤataＦrame既有行索引也有列索引，它可以被看作由Ｓeries组成的字典（共同用一个索引）
##### 创建
创建ＤataＦrame的方法很多，最常用的是一种直接传入一个由等长列表或numpy数组组成的字典
``` python
data = {'state':['ohio', 'ohio', 'ohio', 'nevada', 'nevada'], 
        'year' = ['2000', '2001', '2002', '2001', '2002'], 
        'pop' = [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame                      # -->    pop   state  year
                                                                 0  1.5    ohio  2000
                                                                 1  1.7    ohio  2001
                                                                 2  3.6    ohio  2002
                                                                 3  2.4  nevada  2001
                                                                 4  2.9  nevada  2002
# DataFrame会自动加上索引，且全部列会被有序排列
# 如果指定了列序列，则ＤataＦrame的列就会被按照指定顺序进行排列
DataFrame(data, columns = ['year', 'state', 'pop'])

# 如果传入的列在数据中找不到，就会产生ＮaＮ
frame2 = DataFrame(data, columns = ['year', 'state', 'pop', 'debt'],
                         index = ['one', 'two', 'three', 'four', 'five'])
frame2                     # -->       year   state  pop  debt
                                 one   2000    ohio  1.5   NaN
                                 two   2001    ohio  1.7   NaN
                                 three 2002    ohio  3.6   NaN
                                 four  2001  nevada  2.4   NaN
                                 five  2002  nevada  2.9   NaN
frame2.columns             # --> Index([year, states, pop, debt], dtype = object)
```
##### 索引          
通过类似字典标记的方式或属性的方式，可以将DataFrame的列获取为一个Ｓeries
``` python
frame2['state']
frame2.year                # 注意，返回的Ｓeries拥有原DataFrame相同的索引，且其name属性也已经被相应的设置好了。

# 列可以通过赋值的方式进行修改
frame2['debt'] = 16.5                         # 可以给空的debt列赋上一个标量
frame2['debt'] = np.arange(5)                 # 也可以赋上一组值

# 将列表或数组

# 对行(索引字段)
frame2.ix['three']         # --> year   2002
                                 state  ohio
                                 pop     3.6
                                 debt    NaN
```
##### 
#### 索引对象

### 2-基本功能
#### 重新索引
#### 丢弃指定轴上的项
#### 索引、选取和过滤
#### 算术运算和数据对齐
#### 函数应用和映射
#### 排序和排名
#### 带有重复值的轴索引

### 3-汇总和计算描述统计
#### 相关系数与协方差
#### 唯一值、值计数以及成员资格

### 4-处理缺失数据
#### 滤除缺失数据
#### 填充缺失数据

### 5-层次化索引
#### 重排分级顺序
#### 根据级别汇总统计
#### 使用DataFrame的列

### 6-其他有关Pandas的话题
#### 整数索引
#### 面板数据




matplotlib
------------------






scipy
--------------------









