# 21年春季培训-PaddleHub介绍与使用课后作业

## 题目1：打印Paddle、PaddleHub版本及模型信息


```python
!cp "/home/aistudio/thu_news/base_nlp_dataset.py" /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlehub/datasets/base_nlp_dataset.py
import paddle
print(paddle.__version__) #paddle的版本号
```

    2.0.1



```python
import paddlehub as hub
print(hub.__version__) #paddlehub的版本号
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    2.0.4



```python
!hub list #模型信息
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    +-------------------------+--------------------------------------------------+
    |       ModuleName        |Path                                              |
    +-------------------------+--------------------------------------------------+
    |          ernie          |/home/aistudio/.paddlehub/modules/ernie           |
    +-------------------------+--------------------------------------------------+
    


## 题目2：打印THUCNews验证集前3条、后3条数据


```python
!tar -xzvf data/thu_news.tar.gz
```

    tar (child): data/thu_news.tar.gz: Cannot open: No such file or directory
    tar (child): Error is not recoverable: exiting now
    tar: Child returned status 2
    tar: Error is not recoverable: exiting now



```python
!head -4 thu_news/valid.txt
```

    text_a	label
    直击特里勾手助小牛反超神鸟发威火箭仍处劣势　　新浪体育讯　北京时间4月16日消息，火箭今天迎来常规赛的收官战。客场挑战达拉斯小牛的比赛将关系到火箭队最终的排名，目前西部的竞争仍然非常激励。尤其是西部第二到第七的六支球队将在今天展开捉对厮杀的情况下，黄蜂、小牛、开拓者以及马刺都有可能是火箭队的下一个对手。以下为本场比赛的精彩瞬间——　　第四节比赛还剩下8分多钟，姚明重新回到场上，但是小牛队的霍里随后在右翼接队友传球后上演一记三分跳投，虽然身体动作已经变形，但是他仍然将球命中，随后基德在弧顶处的一记三分跳投帮助小牛终于将比分反超，火箭在进攻中直接打不开局面，而小牛更是利用特里空切后的一记勾手将比分反超至4分，包括库班在内的所有小牛现场球迷都站起来振臂高呼，而此时阿德尔曼还在场边低头深沉的思索中。　　比赛还剩下最后5分多钟，洛瑞带球突破中被小牛队员犯规，洛瑞来不及刹车让自己直接撞到了观众席上，右腿疼痛难忍的他脸都几乎变形，结果他两罚两中，火箭队还落后两分。随后，洛瑞将球直接传给此时已经回到场上的阿泰，野兽此时将球稳下，并没有急于进攻，随后他顺势将球塞给兰德里，神鸟利用对方站位空隙直接杀到篮下，上篮成功，火箭将比分终于追至80平。场上局面对于火箭来说绝对是不容有失。　　(sabrina)	体育
    北京网购年消费额112亿元　　商报讯(记者吴文治)昨日，淘宝网发布的《2009-2010年度中国网购热门城市报告》显示，北京年度网购消费力达112.5亿元，与上海相差近62亿元，位列十大热门消费力城市第二位。此外，男性网购的消费金额高出女性，与“女性是网购主力军”的传统观念不符。　　淘宝公布的报告显示，中国网购消费力十大城市分别是上海、北京、深圳、杭州、广州、南京、苏州、天津、温州和宁波。主要集中在以江浙沪为主的长三角地区、以广深为主的珠三角地区和以北京为主的京津地区。北京年度网购112.5亿元的消费额，占国内城市网购消费额的5.6%。　　中国网购消费力十大城市的消费金额性别来源比例中，男性占比超过了女性。前者占比达到53.5%，后者则为46.5%。不过，在成交人数、成交笔数等关键数据上显示，女性消费者均高于男性。此外，在十大网购热门城市中，25岁-34岁的群体成为网购消费的主力军，占比达到62.49%。	科技
    世界首辆飞行汽车成功试飞(附图)　　新快报讯据美国《时代杂志》等媒体19日报道，世界首辆飞行汽车本月初在美国实现了首飞，降落后只需按一个按钮就可将机翼折叠，驶上高速公路。　　据报道，美国特拉弗吉亚公司研发的这辆名为“特拉弗吉亚过渡”的飞行汽车的空中巡航时速可达185公里，它还可以在道路上行驶，可以安全地存放在车库里。公司称，汽车可以用一箱汽油在空中飞行643公里。与此同时，飞行汽车加油时只需驶入最近的加油站，向油箱里加入无铅汽油即可。人们先前也曾验证过其它飞行汽车，但特拉弗吉亚公司研发的飞行汽车是第一辆拥有可折叠机翼的、并能无缝隙地实现从空中降落到公路的汽车。不过，这辆车的门槛还有点高，车的价格达13.9万英镑（约合135万元人民币）。	社会



```python
!tail -3 thu_news/valid.txt
```

    事业测试：你工作易受他人干扰吗(图)　　独家撰稿：苑椿　心理测试征稿启事 欢迎关注新浪星座微博　　办公室永远是个龙蛇混杂、藏龙卧虎的地方，你永远不知道一张张面具底下会是怎样的脸庞，你是否还傻傻的对别人的话听之任之，完全搞乱了自己工作的步调？还是笃定的坚守阵地，从未被谣言动摇分毫？赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    趣味测试：你怎么红红火火过春节(图)　　独家撰稿：岚　心理测试征稿启事 欢迎关注新浪星座微博　　红红火火过大年啦，每年的此时你都会如何度过呢？是跟家人爱人在一起还是跟朋友兄弟外出欢聚呢？亦或背起行囊远离嘈杂，无论如何，总会有一种适合你的方式，赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    人际测试：你的人际磁场强大吗(图)　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。	星座


## 题目3：完整跑通基于PaddleHub的新闻文本分类任务

### Step1 选择模型


```python
import paddlehub as hub

model = hub.Module(name='ernie', version='2.0.2', task='seq-cls', num_classes=14)

```

    [2021-04-14 00:24:48,432] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1303: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1303: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


### Step2 数据处理


```python
import pandas as pd
df =  pd.read_table("thu_news/test.txt")
df[["text_a", "label"]] = df[["label", "text_a"]]
df.to_csv('thu_news/test-1.csv', index=0, sep='\t')
```


```python
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset

class MyDataset(TextClassificationDataset):
    base_path = "/home/aistudio/thu_news/"
    label_list = ['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']

    def __init__(self, tokenizer, max_seq_len: int=256, mode:str='train'):
        if mode == 'train':
            data_file = 'train-1.csv'
        elif mode == 'test':
            data_file = 'test-1.csv'
        else:
            data_file = 'valid-1.csv'
        
        super().__init__(
            base_path=self.base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=self.label_list,
            is_file_with_header=True,
        )

tokenizer = model.get_tokenizer()
```

    [2021-04-14 00:24:54,967] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-1.0/vocab.txt



```python
train_dataset = MyDataset(tokenizer)
valid_dataset = MyDataset(tokenizer, mode='valid')
test_dataset = MyDataset(tokenizer, mode='test')
```

### Step3 选择优化策略和运行配置


```python

optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt', use_gpu=True)

# trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset)

# 在测试集上评估当前训练模型
#trainer.evaluate(test_dataset, batch_size=32)
```

    [2021-04-14 00:25:07,490] [ WARNING] - PaddleHub model checkpoint not found, start from scratch...


### Step4 预测


```python
data = [
    # 房产
    ["昌平京基鹭府10月29日推别墅1200万套起享97折　　新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。　　京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。"],
    # 游戏
    ["尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。　　这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。　　片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。"],
    # 体育
    ["罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪　　记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。　　拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。"],
    # 教育
    ["新总督致力提高加拿大公立教育质量　　滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。　　约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。"]
]

label_list=['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']
label_map = { 
    idx: label_text for idx, label_text in enumerate(label_list)
}

model = hub.Module(
    name='ernie',
    task='seq-cls',
    load_checkpoint='./ckpt/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=False)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
```

    [2021-04-14 00:25:07,500] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1303: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1303: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-04-14 00:25:09,031] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-1.0/vocab.txt


    Data: 昌平京基鹭府10月29日推别墅1200万套起享97折　　新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。　　京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。 	 Lable: 科技
    Data: 尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。　　这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。　　片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。 	 Lable: 科技
    Data: 罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪　　记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。　　拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。 	 Lable: 科技
    Data: 新总督致力提高加拿大公立教育质量　　滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。　　约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。 	 Lable: 科技

