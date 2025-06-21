# 🧠 CBOW 模型代码理解文档

本项目实现了一个基于 CBOW（Continuous Bag-of-Words）的文本分类模型。以下为对核心类、方法及训练流程的详细理解总结，共 25 个关键问答，适合用于复习、讲解或代码维护。

---

## 代码理解问答总结

### Vocabulary 类

1. **mask_token 对应的索引赋值给哪个属性？**  
   答：`self.mask_index`

2. **lookup_token 方法中未登录词返回什么？**  
   答：`self.unk_index`（前提是 `unk_index >= 0`）

3. **add_many 方法实际调用了哪个方法？**  
   答：`add_token`

---

### CBOWVectorizer 类

4. **当 vector_length < 0 时，向量长度为？**  
   答：`self._max_seq_length` 的长度

5. **from_dataframe 方法会遍历哪两列？**  
   答：`context` 和 `target`

6. **out_vector[len(indices):] 的填充值？**  
   答：`self.cbow_vocab.mask_index`

---

### CBOWDataset 类

7. **_max_seq_length 是如何得到的？**  
   答：所有 context 列的最大 `len(context)` 值

8. **set_split 方法从 _lookup_dict 选出哪些字段？**  
   答：`split_data` 和 `split_index`

9. **__getitem__ 中 y_target 如何获得？**  
   答：通过查找 `target` 列中的 token 得到

---

### CBOWClassifier 模型结构

10. **x_embedded_sum 的计算方式？**  
    答：`embedding(x_in).sum(dim=1)`

11. **输出层 fc1 的 out_features 等于？**  
    答：`num_classes` 参数的值

---

### 训练流程

12. **generate_batches 使用了哪个 PyTorch 类？**  
    答：`DataLoader`

13. **classifier.train() 会启用哪些模式？**  
    答：训练模式 和 Dropout 模式

14. **反向传播前要执行什么操作清空梯度？**  
    答：`optimizer.zero_grad()`

15. **compute_accuracy 中如何获取预测类别？**  
    答：使用 `torch.max()` 方法获取 `y_pred_indices`

---

### 训练状态管理

16. **make_train_state 中 early_stopping_best_val 的初始值？**  
    答：`inf`

17. **连续多少次验证损失未下降会触发早停？**  
    答：`early_stopping_criteria`

18. **验证损失下降时 early_stopping_step 重置为？**  
    答：`0`

---

### 设备与随机种子

19. **set_seed_everywhere 中 CUDA 相关设置？**  
    答：`torch.cuda.manual_seed_all(seed)`

20. **args.device 的值依据什么确定？**  
    答：`torch.cuda.is_available()`

---

### 推理与测试

21. **get_closest 函数如何排除自身词？**  
    答：判断 `word == target_word`，使用 `continue`

22. **测试集评估时要调用哪个方法？**  
    答：`model.eval()`

---

### 关键参数

23. **CBOWClassifier 的 padding_idx 默认值？**  
    答：`0`

24. **控制词向量维度的参数名？**  
    答：`embedding_size`

25. **ReduceLROnPlateau 触发条件是验证损失？**  
    答：**增加**（未减少）
