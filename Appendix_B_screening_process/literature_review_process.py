import graphviz

# 创建Digraph对象，优化布局参数
dot = graphviz.Digraph('LiteratureReview', format='png',
                       graph_attr={
                           'rankdir': 'TB',
                           'fontsize': '10',  # 缩小字体
                           'fontname': 'Arial',
                           'bgcolor': 'transparent',
                           'margin': '0.2',  # 减小边距
                           'nodesep': '0.15',  # 减小节点间距
                           'ranksep': '0.25'  # 减小层级间距
                       },
                       node_attr={
                           'fontname': 'Arial',
                           'shape': 'box',
                           'style': 'filled',
                           'fillcolor': '#e7f3ff',
                           'color': '#4a90e2',
                           'penwidth': '1.1',  # 减小线宽
                           'margin': '0.1'  # 减小节点内边距
                       },
                       edge_attr={
                           'color': '#4a90e2',
                           'penwidth': '1.1',
                           'fontname': 'Arial',
                           'fontsize': '9'  # 缩小边标签字体
                       })

# ---------- 紧凑型节点样式 ----------
compact_style = {
    'fontsize': '10',
    'margin': '0.08,0.03'  # 水平,垂直内边距
}

search_db_style = {
    'shape': 'cylinder',
    'fillcolor': '#d0e4ff',
    'color': '#2c7bb6',
    'penwidth': '1.3',
    **compact_style
}

process_step_style = {
    'shape': 'rect',
    'fillcolor': '#e7f3ff',
    'color': '#4a90e2',
    'penwidth': '1.3',
    **compact_style
}

intermediate_result_style = {
    'shape': 'note',
    'fillcolor': '#94c0ff',
    'color': '#2c7bb6',
    'penwidth': '1.3',
    **compact_style
}

final_result_style = {
    'shape': 'note',
    'style': 'filled,rounded',
    'fillcolor': '#2c7bb6',
    'color': '#1b4f72',
    'penwidth': '1.5',
    'fontsize': '11',
    'fontcolor': 'white',
    'margin': '0.1'
}

# ---------- 搜索查询节点（更紧凑）----------
dot.node('Query',
         '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">'
         '<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="11" FACE="Arial-Bold">Search Query</FONT></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">(metaverse OR "virtual world")<BR/>'
         'AND ("multi-agent system" OR MAS)</FONT></TD></TR>'
         '</TABLE>>',
         shape='plaintext')

# ---------- 数据库源（横向排列更紧凑）----------
with dot.subgraph(name='cluster_databases') as s:
    s.attr(label='Digital Libraries',
           style='rounded,filled',
           fillcolor='#f1f8ff',
           color='#c6d9f1',
           fontname='Arial-Bold',
           fontsize='10',
           margin='8',
           rank='same')  # 强制同级排列

    # 使用更简洁的标签
    s.node('ACM', 'ACM\n76', **search_db_style)
    s.node('IEEE', 'IEEE\n181', **search_db_style)
    s.node('Scopus', 'Scopus\n67', **search_db_style)
    s.node('SD', 'SD\n182', **search_db_style)
    s.node('Springer', 'Springer\n125', **search_db_style)

# ---------- 初始结果 ----------
dot.node('Total1', 'Initial\n631', **intermediate_result_style)

# 连接数据库到初始结果（更紧凑的边）
for src in ['ACM', 'IEEE', 'Scopus', 'SD', 'Springer']:
    dot.edge('Query', src, style='dashed', arrowhead='none', minlen='1')  # 减小最小长度
    dot.edge(src, 'Total1', color='#2c7bb6')

# ---------- 处理步骤（合并部分节点）----------
steps = [
    ('P1', 'Merge & Clean\n-73', 'R1', '558'),
    ('P2', 'Remove Duplicates\n-12', 'R2', '546'),
    ('P3', 'Filter Records\n-214', 'R3', '332'),
    ('P4', 'Screening\n-186', 'R4', '146'),
    ('P5', 'Remove Pre-2015\n-21', 'R5', '125')
]

prev_result = 'Total1'
for pid, ptext, rid, rtext in steps:
    dot.node(pid, ptext, **process_step_style)
    dot.node(rid, rtext, **intermediate_result_style)
    dot.edge(prev_result, pid)
    dot.edge(pid, rid)
    prev_result = rid

# ---------- 最终步骤 ----------
dot.node('Snow', 'Snowball\n+12', **process_step_style)
dot.node('Final', 'Final\n137', **final_result_style)

dot.edge('R5', 'Snow')
dot.edge('Snow', 'Final')

# 渲染为PNG（启用紧凑引擎）
dot.render('literature_review_process',
           view=True,
           cleanup=True,
           format='png',
           engine='dot')  # 使用dot引擎以获得最佳布局

print("优化后的文献综述流程图已保存为 'literature_review_process.png'")