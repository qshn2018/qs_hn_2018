Python基础
=======================
0-写在前面
-------------------------------
#### Python vs 其他语言
相比其他语言，Python最大的不同就是需要缩进！！！

#### 注释
``` python
'''
这是多行注释

通常写在整个python文件的开头
作说明性介绍

或者用于需要多次换行的注释
'''
                           # 这是单行注释，用#开头
```

1-基本数据类型
-------------------------------
#### 整数
``` python
1                          # 正数
-1                         # 负数
0                          # 零
```
#### 小数
``` python
3.14
1.0                        # 只要有小数点就是小数
1.23e9                     # e记法，代表1.23x10
```
#### 字符串
``` python
'hello'                    # 字符串需要用引号括起来，可以用单引号，注意：引号不作为字符串的内容
"hello"                    # 也可以用双引号，但是必须成对（不可一单一双），且是英文的
'''hello'''                # 三引号也可以，但是通常作为注释使用
```
但是，字符串中需要带有引号时的解决方案：  
- 内容中带有单引号的用双引号括。  
- 内容中带有双引号的用单引号括。  
- 内容中单双引号都存在的用三引号括。  
- 转义字符：`\'`代表单引号,`\"`代表双引号  
- `r'字符串内容'`：这种写法会默认不转义  

#### bool
``` python
True                       # 代表 真
False                      # 代表 假

1 < 2                      # 逻辑表达式会自动计算出bool值
```
#### None
``` python
None                       # None代表空值，代表什么也没有。空值不是0,0有数学意义，而None没有
```

运算操作
-------------------------------------
#### 数学运算
``` python
1 + 1                      # 加法
3.14 - 1.23                # 减法
2.3 * 4                    # 乘法
5 / 2                      # 除法
5 // 2                     # 地板除
5 % 2                      # 取余
5 ** 2                     # 乘方
```
#### 比较运算
``` python
1 == 1                     # 等于
1 != 1                     # 不等于
1 > 2                      # 大于
1 < 2                      # 小于
1 <= 2                     # 小于等于
1 >= 2                     # 大于等于
```
``` python
not 1                      # 非
1 and 1                    # 与
1 or 2                     # 或
```
#### 优先级


变量&常量
----------------------------------
#### 变量
Python中的变量就是一个标签，是一个名字，用于”指向“数据，而不是存储数据，这里区别于别的编程语言。  
Python中的变量不需要先声明，这里区别于别的编程语言。  
变量可以指向在此之前及之后提到的所有数据类型和数据结构。  

变量的命名规则：
- 由字母，数字，下划线构成
- 不能由数字开头
- 大小写敏感

变量的赋值操作：  
给变量赋值，就是让变量名指向数据，指向所赋值的数据  

``` python
num = 1                    # 指向整数
number = 3.14              # 指向小数
name = 'Bob'               # 指向字符串
i = True                   # 指向bool值
x = None                   # 指向空值
```
#### 常量
常量即为不变的量，但是在Ｐython中没有办法能让其不变  
常量和变量无二致，只是约定将变量名全部大写来提醒这是常量不要轻易改变  



输入输出
---------------------------------
#### print()



#### input()







数据结构
---------------------------------
#### list
#### tuple
#### dict
#### set







### 函数
--------------------------------------
#### 函数的调用
#### 函数的定义
--匿名函数
#### 函数的参数
--偏函数
#### 高阶函数
#### 返回函数
--闭包
#### 装饰器










### 模块
----------------------------------------
#### 导入模块
#### 常用内建模块
--random
#### 常用第三方模块
--numpy
--pandas
--matplotlib
--scipy
--tensorflow






### 类&对象
----------------------------------------






### IO
------------------------------------------

