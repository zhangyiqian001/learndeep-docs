# Introduction

## 目标

1. 整理经典、前沿深度学习论文（翻译、注释）以供讨论学习
2. 复现相关论文代码，对照学习。
3. 一个专注于模型实现的框架:[learndeep](https://github.com/zhangyiqian001/learndeep.git)
4. 为提高效率，部分论文通过AI总结生成后，人工校对。

## TODO

## 贡献

### 预备知识

- [gitbook文档](https://chrisniael.gitbooks.io/gitbook-documentation/content/index.html)
- [markdownlint（vs 插件）: markdown开发规范](https://github.com/DavidAnson/markdownlint)


### docker环境

项目地址: https://github.com/billryan/docker-gitbook

```shell
# serve
docker run --rm -v "$PWD:/gitbook" -p 4000:4000 billryan/gitbook gitbook serve
# build
docker run --rm -v "$PWD:/gitbook" -p 4000:4000 billryan/gitbook gitbook build
```

### 开发规范
```markdown
# 摘要

## 方法

## 实验

## 相关工作

## QA

## 引用

```