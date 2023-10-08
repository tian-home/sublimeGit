def BI_varRank(s_var):
    # s_var = r'KPI,创维(SKYWORTH),夏普(SHARP),LG,TCL,海信(Hisense),小米(MI),华为(HUAWEI),荣耀(HONOR),三星(SAMSUNG),索尼(SONY),飞利浦(PHILIPS),左侧均无'

    l_var = s_var.split(",")

    s_result = r'Table.AddColumn(更改的类型1, "Rank_{}", each if [{}] = "{}" then 1'.format(l_var[0], l_var[0], l_var[1])

    for i, v in enumerate(l_var):
        # print(i, v)
        if i > 1:
            s_result += '\n else if [{}] = "{}" then {} '.format(l_var[0], l_var[i], i)
    s_result = s_result + r' else null, type number)'
    print(s_result)

"""run"""
# s_var = r'KPI,创维(SKYWORTH),夏普(SHARP),LG,TCL,海信(Hisense)'
# BI_varRank(s_var)

"""origin"""
# s_var = r'KPI,创维(SKYWORTH),夏普(SHARP),LG,TCL,海信(Hisense),小米(MI),华为(HUAWEI),荣耀(HONOR),三星(SAMSUNG),索尼(SONY),飞利浦(PHILIPS),左侧均无'
# l_var = s_var.split(",")
# s_result = r'Table.AddColumn(更改的类型1, "Rank_{}", each if [{}] = "{}" then 1'.format(l_var[0], l_var[0], l_var[1])
# for i, v in enumerate(l_var):
#     # print(i, v)
#     if i > 1:
#         s_result += '\n else if [{}] = "{}" then 2 '.format(l_var[0], l_var[i])
# s_result = s_result + r' else null, type number)'
# print(s_result)

