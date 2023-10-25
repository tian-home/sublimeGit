def BI_varRank(s_var):
    # s_var = r'KPI,创维(SKYWORTH),夏普(SHARP),LG,TCL,海信(Hisense),小米(MI),华为(HUAWEI),荣耀(HONOR),三星(SAMSUNG),索尼(SONY),飞利浦(PHILIPS),左侧均无'

    l_var_1 = s_var.split(",")[:2]
    l_var_2 = s_var.split(",")[2:]

    s_result = r'=Table.AddColumn(更改的类型1, "Rank_{}", each if [{}] = "{}" then 1'.format(l_var_1[0], l_var_1[1], l_var_2[0])

    for i, v in enumerate(l_var_2):
        # print(i, v)
        if i > 0:
            s_result += '\n else if [{}] = "{}" then {} '.format(l_var_1[1], l_var_2[i], i+1)
    s_result = s_result + r' else null, type number)'
    print(s_result)

"""run"""
# s_var = r'KPI,创维(SKYWORTH),夏普(SHARP),LG,TCL,海信(Hisense),小米(MI),华为(HUAWEI),荣耀(HONOR),三星(SAMSUNG),索尼(SONY),飞利浦(PHILIPS),左侧均无'
s_var = r'city,Q4城市 SAR,北京,上海,广州,深圳,沈阳,天津,西安,济南,南京,杭州,武汉,福州,成都'
# s_var = r'PurchaseChannel,购买渠道 SAR,实体店,网上商城,其他'
BI_varRank(s_var)

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

