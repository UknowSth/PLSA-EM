from model import PLSA

em = PLSA('small.txt',topk=50,k=10,iteration=45)
em.load()
em.initialize()
em.iteration = 45 # 可以更改迭代次数
em.update()
em.printTopK(20)

# 将优化后的参数保存下来
em.save_para(path='small-output.json') 

# 再次定义模型 加载参数
em = PLSA()
em.load_para(path='small-output.json')

# 可视化
em.show_graph(50)