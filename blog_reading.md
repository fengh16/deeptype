# Discovering Types for Entity Disambiguation

https://openai.com/blog/discovering-types-for-entity-disambiguation/

We’ve built a system for automatically figuring out which object is meant by a word 一个词代表的物体是什么 by having a neural network decide if the word belongs to each of about 100 automatically-discovered “types” 100个自动找到的类别 (non-exclusive 非排他的 categories). For example, given a sentence like “the prey saw the jaguar cross the jungle”, rather than trying to reason directly whether jaguar means the car, the animal, or something else, the system plays “20 questions” （问20个问题猜出东西是什么） with a pre-chosen set of categories 类似决策树的一种问问题做选择的方式. This approach gives a big boost in state-of-the-art 最先进的 on several entity disambiguation 实体消歧 datasets.

In our training data jaguar refers to the car 70% of the time, the animal 29% of the time, and the aircraft 1% of the time. With our types approach, the possible disambiguations in the first example don’t change a huge amount — apparently the model is ok with jaguars running down the highway — but change hugely in the second — it’s not ok with Jaguars taking a cruise through the jungle.

We achieve 94.88% accuracy on CoNLL (YAGO) (previous state of the arts: 91.50%, 91.70%) and 90.85% on TAC KBP 2010 challenge (previous state of the arts: 87.20%, and 87.70%). Previous methods used distributed representations. Types can go almost all the way on these tasks, as perfect type prediction would give accuracies of 98.6-99%.

## High-level overview

Our system uses the following steps:

1. Extract every Wikipedia-internal link to determine, for each word, the set of conceivable entities it can refer to. For example, when encountering 遇到 the link [jaguar](https://en.wikipedia.org/wiki/Jaguar) in a Wikipedia page, we conclude that https://en.wikipedia.org/wiki/Jaguar is one of the meanings of jaguar.
2. Walk the Wikipedia category tree (using the Wikidata knowledge graph) 用维基数据知识图谱遍历类别树 to determine, for each entity, the set of categories it belongs to 确定每一个实体的类别. For example, at the bottom of https://en.wikipedia.org/wiki/Jaguar_Cars’s Wikipedia page, are the following categories (which themselves have their own categories, such as Automobiles): British hrands、Car brands、Jaguar Cars、Jaguar vehicles
3. Pick a list of ~100 categories to be your “type” system, and optimize over this choice of categories 优化类别选择 so that they compactly 紧凑地 express any entity 用类别表示任何实体. We know the mapping of entities to categories, so given a type system, we can represent each entity as a ~100-dimensional binary vector indicating membership in each category. 用100维的向量表示每一个实体（每一个维度都是一个类别的布尔值）
4. Using every Wikipedia-internal link and its surrounding context 连接和对应的附近的文本, produce training data mapping a word plus context to the ~100-dimensional binary representation of the corresponding entity 把这个词和对应的上下文表示成表示这个实体的100维布尔值表示, and train a neural network to predict this mapping 用神经网络预测这样的表示. This chains together the previous steps 这些步骤: Wikipedia links map a word to an entity 把一个词对应到一个entity, we know the categories for each entity from step 2 了解到每一个entity的类别, and step 3 picked the categories in our type system 在我们的类别系统中选择类别.
5. At test time 测试的时候, given a word and surrounding context 词和对应的上下文, our neural network’s output can be interpreted as the probability that the word belongs to each category 给出属于每一个类别的概率. If we knew the exact set of category memberships, we would narrow down to one entity (assuming well-chosen categories) 如果已经知道了实际的类别，就可以确定出这个实体（假设类别选择很合理）. But instead, we must play a probabilistic 20 questions: use Bayes’ theorem 贝叶斯算法 to calculate the chance of the word disambiguating to each of its possible entities 计算可能的消歧义概率.

## Cleaning the data数据清理

Wikidata’s knowledge graph can be turned into a source of training data for mapping fine-grained 细粒度的 entities to types 把维基的数据用来作为实体到类别的训练数据. We apply its instance of relation recursively 递归地 to determine the set of types for any given entity 决定每一个实体饿类别集合 — for example, any descendent node of the human node has type human 人的子节点（派生）都是人. Wikipedia can also provide entity-to-type mapping through its category link.

Wikipedia-internal link statistics provide a good estimate of the chance a particular phrase refers to some entity. 维基内部的链接给出了一个对于每一个短语是什么实例的估计 However, this is noisy 但是有噪声 since Wikipedia will often link to specific instance of a type 经常会到具体的类型的实例 rather than the type itself 而不是这个类别 (anaphora — e.g. king → Charles I of England) （也就是说，页面里面提到了XXX是一个国王，国王这个词点进去之后会发现并不是“国王”页面，而是这个国王的页面，就把king和这个名字建立起了联系） or link from a nickname (metonymy). This results in an explosion of associated entities (e.g. king has 974 associated entities) and distorted 扭曲的 link frequencies (e.g. queen links to the band Queen 4920 times, Elizabeth II 1430 times, and monarch only 32 times).

The easiest approach is to prune rare links 修剪掉稀有的连接, but this loses information 造成了信息损失. We instead use the Wikidata property graph 属性图 to heuristically 启发式的 turn links into their “generic” meaning 转换成自己的比较泛化的意思, as illustrated below.

## Learning a good type system学习一个好的类型系统

We need to select the best type system and parameters 选择好的类型系统和参数 such that disambiguation accuracy is maximized. There’s a huge number of possible sets of types 类别的集合可选择的有很多, making an exact solution intractable 精确的结果就很难处理. Instead, we use heuristic search 启发式搜索 or stochastic optimization 随机优化 (evolutionary algorithm 进化算法) to select a type system, and gradient descent 梯度下降 to train a type classifier to predict the behavior of the type system 预测类型系统行为.

The receiver operating characteristics curve 工作特征曲线 (ROC) plots how increasing the number of true positives (actual events positively detected) increases with respect to the number of false positives (false alarms caused by being trigger happy) 用true positive和false positive画图. Here the curve has an AUC of 0.75. A classifier that acts randomly will by chance have a straight line ROC curve (dashed line) 随机的是一条直线.

AUC是ROC曲线下面的面积，ROC分析的是二元分类模型，真阳性（TP）：判断为阳性，实际也是阳性；伪阳性（FP）：判断为阳性，实际却是阴性。ROC曲线将假阳性率（FPR）定义为 X 轴，真阳性率（TPR）定义为 Y 轴。其中：TPR：在所有实际为阳性的样本中，被正确地判断为阳性的样本比率；FPR：在所有实际为阴性的样本中，被错误地判断为阳性的样本比率。TPR = TP / (TP + FN)；FPR = FP / (FP + TN)。因为判断过程需要一个阈值，所以每一个阈值对应一个点。

We need to select types that are discriminating 区分 (so quickly whittle down 削减 the possible set of entities), while being easy to learn 易于学习 (so surrounding context is informative for a neural network to infer that a type applies 上下文用来推断应用的类别的信息). We inform our search with two heuristics 两种启发式: learnability 可学习性 (average of area under the curve [AUC] scores of a classifier trained to predict type membership 用AUC分数), and oracle accuracy (how well we would disambiguate entities if we predicted all types perfectly 如果类别全都对了，结果的预测会怎样).

## Type system evolution

For each possible type a different binary classifier is trained 每一个类别训练一个分类器: here we decide whether a token refers to an “Animal” article. （IsAnimal?）

We train binary classifiers to predict membership in each of the 150,000 most common types in our dataset 15万个类别中的成员, given a window of context. The area under the curve (AUC) of the classifier becomes the “learnability score” for that type. High AUC means it’s easy to predict this type from context; poor performance can mean we have little training data 数据不够 or that a word window isn’t terribly helpful 单词窗口不太有用 (this tends to be true for unnatural categories like ISBNs 人造的类别是如此). Our full model takes several days to train, so we instead use a much smaller model as a proxy in our “learnability score”, which takes only 2.5s to train.

We can now use these learnability scores and count statistics to estimate the performance of a given subset of types as our type system. Below you can run the Cross Entropy Method to discover types in your browser. Note how changing sample size and penalties affects the solution.

To better visualize what parts of the type system design are easy and hard, we invite you to try your hand at designing your own below. After choosing a high-level domain you can start looking at ambiguous examples. The possible answers are shown as circles on the top row, and the correct answer is the colored circle (hover to see its name). The bottom row contains types you can use. Lines connecting the top to the bottom row are inheritance relations. Select the relations you want. Once you have enough relations to separate the right answer from the rest, the example is disambiguated.

## Neural type system

Using the top solution from our type system optimization, we can now label data from Wikipedia using labels generated by the type system. Using this data (in our experiments, 400M tokens for each of English and French), we can now train a bidirectional LSTM 双向LSTM to independently predict all the type memberships for each word 独立预测每一个单词的所有类别成员. On the Wikipedia source text, we only have supervision on intra-wiki links 内部连接, however this is sufficient to train a deep neural network to predict type membership with an F1 of over 0.91.

One of our type systems, discovered by beam search, includes types such as Aviation, Clothing, and Games (as well as surprisingly specific ones like 1754 in Canada — indicating 1754 was an exciting year in the dataset of 1,000 Wikipedia articles it was trained on); you can also view the full type system.

## Inference 推论

Predicting entities in a document usually relies on a “coherence” 一致性 metric 度量 between different entities 不同实体之间的一致性度量, e.g. measuring how well each entity fits with each other 测量每一个实体的匹配程度, which is O(N^2) in the length of the document. Instead, our runtime is O(N) as we need only to look up each phrase in a trie 单词树里面查找短语 which maps phrases to their possible meanings. We rank each of the possible entities 给每一个可能的实体排序 according to the link frequency seen in Wikipedia 用wiki里面的频率排序, refined by weighting each entity by its likelihood under the type classifier 在类型下的可能性进行加权优化. New entities can be added just by specifying their type memberships 通过说明类型成员添加新的实体 (person, animal, country of origin, time period, etc..).

## Next steps

Our approach has many differences to previous work on this problem. We’re interested in how well end-to-end learning of distributed representations performs in comparison to the type-based inference we developed here. The type systems here were discovered using a small Wikipedia subset; scaling to all of Wikipedia could discover a type system with broad application. We hope you find our code useful!

If you’d like to help push research like this forward, please apply to OpenAI!

Authors

Jonathan Raiman