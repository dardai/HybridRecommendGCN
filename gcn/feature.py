# -*- coding: utf-8 -*-

import csv
import numpy as np

from enum import Enum, IntEnum

from databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, getEnumValue

import logging

class CourseType(Enum):
    CourseType_None = 0
    CourseType_Vedio = 1
    CourseType_RichMedia = 2
    CourseType_Text = 3
    CourseType_Scrom = 4


class CourseDiffer(IntEnum):
    CourseDiffer_None = 0
    CourseDiffer_2017Lion = 1
    CourseDiffer_A1Type = 2
    CourseDiffer_A3Type = 3
    CourseDiffer_A4Type = 4
    CourseDiffer_B4Type = 5
    CourseDiffer_C1Type = 6
    CourseDiffer_C2TypeCartoon = 7
    CourseDiffer_C2TypeHigh = 8
    CourseDiffer_EHS = 9
    CourseDiffer_Excel = 10
    CourseDiffer_TCP = 11
    CourseDiffer_UXTest = 12
    CourseDiffer_YiDaiYiLu = 13
    CourseDiffer_ShangHaiZiMaoQu = 14
    CourseDiffer_YeWuGaiKuang = 15
    CourseDiffer_GeRenYeWu = 16
    CourseDiffer_GeRenZhengXin = 17
    CourseDiffer_GeRenLiCaiCase = 18
    CourseDiffer_GeRenLiCaiBase = 19
    CourseDiffer_GeRenLiCaiShiWu = 20
    CourseDiffer_GeRenDaiKuan = 21
    CourseDiffer_GeRenStyleColor = 22
    CourseDiffer_TongyongTitleStudy = 23
    CourseDiffer_YunPingTai = 24
    CourseDiffer_InternetPlus = 25
    CourseDiffer_InternetEnv = 26
    CourseDiffer_ImproveCapacity = 27
    CourseDiffer_ChildRead = 28
    CourseDiffer_ManageHumanResource = 29
    CourseDiffer_CompanyCulture = 30
    CourseDiffer_CompanyKonwledge = 31
    CourseDiffer_ChuanShiJingDian = 32
    CourseDiffer_InsureOnePoint = 33
    CourseDiffer_InsureBase = 34
    CourseDiffer_InsureBaseKnowledge = 35
    CourseDiffer_InsureNDA = 36
    CourseDiffer_InsurePropertyManage = 37
    CourseDiffer_InsureKeJi = 38
    CourseDiffer_SellInsure = 39
    CourseDiffer_InsureCounselorServer = 40
    CourseDiffer_BeQualifiedPartyMember = 41
    CourseDiffer_MiniClassForDangJian = 42
    CourseDiffer_DangZhangKnowledge = 43
    CourseDiffer_FirstClassForJionParty = 44
    CourseDiffer_Other = 45
    CourseDiffer_NeiXunShi = 46
    CourseDiffer_Identify = 47
    CourseDiffer_InnovateFromFunc = 48
    CourseDiffer_InnovateBeginBussiness = 49
    CourseDiffer_InnovateWithCompany = 50
    CourseDiffer_FuncOfInnovate = 51
    CourseDiffer_QuKuaiLian = 52
    CourseDiffer_BoYiYuJueCe = 53
    CourseDiffer_YaLiManage = 54
    CourseDiffer_CanKaoBaoJia = 55
    CourseDiffer_FanXiQianBase = 56
    CourseDiffer_PingPaiZhanLue = 57
    CourseDiffer_TastingWine = 58
    CourseDiffer_BussinessModel = 59
    CourseDiffer_BussinesEtiquette = 60
    CourseDiffer_TaiBaoPersonManageSys = 61
    CourseDiffer_YangHangFanXiQian = 62
    CourseDiffer_Luxury = 63
    CourseDiffer_HongGuanJingJi = 64
    CourseDiffer_KeHuServer = 65
    CourseDiffer_DuiChongJiJin = 66
    CourseDiffer_MiniCompanyJinRong = 67
    CourseDiffer_ShiChangYingXiao = 68
    CourseDiffer_MiniClassMegagame = 69
    CourseDiffer_MiniClassMegagameSend = 70
    CourseDiffer_OperateSkill = 71
    CourseDiffer_EducatePsychology = 72
    CourseDiffer_CultureAndEthic = 73
    CourseDiffer_NewEmployee = 74
    CourseDiffer_NewEmployeeProject = 75
    CourseDiffer_NewCountryIntro = 76
    CourseDiffer_NewCountryDetail = 77
    CourseDiffer_DoNewMedia = 78
    CourseDiffer_DailyRun = 79
    CourseDiffer_BookOfChangesClass = 80
    CourseDiffer_MorningEveningMeet = 81
    CourseDiffer_BestPractice = 82
    CourseDiffer_CounterInHeart = 83
    CourseDiffer_CaseTeach = 84
    CourseDiffer_EachPersonIsVIP = 85
    CourseDiffer_GoodMoodIntrest = 86
    CourseDiffer_GoodMoodSuccess = 87
    CourseDiffer_GoodMoodPositiveThink = 88
    CourseDiffer_LawPublicity = 89
    CourseDiffer_ActivityRunAnsower = 90
    CourseDiffer_ActivityRunLive = 91
    CourseDiffer_ActivityRunReadBook = 92
    CourseDiffer_PlayGolf = 93
    CourseDiffer_SceneInterview = 94
    CourseDiffer_ETextbook = 95
    CourseDiffer_CyclopediaEntry = 96
    CourseDiffer_BuildKnowledgeLib = 97
    CourseDiffer_KnowledgeBrochure = 98
    CourseDiffer_ManageSkill = 99
    CourseDiffer_ManageTheory = 100
    CourseDiffer_ManageStudyNearBody = 101
    CourseDiffer_ProfessionDevelopSkill = 102
    CourseDiffer_ProfessionMorality = 103
    CourseDiffer_WorkplaceStudyNearBody = 104
    CourseDiffer_FreeTradeArea = 105
    CourseDiffer_TeaTalkNewWord = 106
    CourseDiffer_AdministrationExplain = 107
    CourseDiffer_VedioDesign = 108
    CourseDiffer_SecurityInvest = 109
    CourseDiffer_Cource = 110
    CourseDiffer_MakeCourse = 111
    CourseDiffer_CourseDesignDevelopOne = 112
    CourseDiffer_CourseDesignDevelopTwo = 113
    CourseDiffer_BaseMeasureManage = 114
    CourseDiffer_QualificationAuthentication = 115
    CourseDiffer_RunInternalPiloting = 116
    CourseDiffer_RunManage = 117
    CourseDiffer_TongYongManage = 118
    CourseDiffer_SectionShow = 119
    CourseDiffer_FinancialReform = 120
    CourseDiffer_FinancialDerivative = 121
    CourseDiffer_BankLawAndSkill = 122
    CourseDiffer_BankService = 123
    CourseDiffer_BankCounterAcrobatics = 124
    CourseDiffer_BankCounterSkill = 125
    CourseDiffer_BankManage = 126
    CourseDiffer_SellSkill = 127
    CourseDiffer_RetailCreditWork = 128
    CourseDiffer_InterviewSkill = 129
    CourseDiffer_TopSellor = 130
    CourseDiffer_Project = 131
    CourseDiffer_Leadership = 132
    CourseDiffer_RiskCompliance = 133
    CourseDiffer_TopSellCase = 134
    CourseDiffer_TopManager = 135
    CourseDiffer_GoldInvest = 136


def transformCourseType(courseType):
    logging.warning("运行日志：转换课程类型")
    result = CourseDiffer.CourseDiffer_None

    if courseType.encode("utf-8") == '2017狮王争霸':
        result = CourseDiffer.CourseDiffer_2017Lion
    elif courseType.encode("utf-8") == 'A1类： PPT翻页':
        result = CourseDiffer.CourseDiffer_A1Type
    elif courseType.encode("utf-8") == 'A3类： PPT讲授':
        result = CourseDiffer.CourseDiffer_A3Type
    elif courseType.encode("utf-8") == 'A4类： 图文设计':
        result = CourseDiffer.CourseDiffer_A4Type
    elif courseType.encode("utf-8") == 'B4类： 互动营销':
        result = CourseDiffer.CourseDiffer_B4Type
    elif courseType.encode("utf-8") == 'C1类： 讲师授课':
        result = CourseDiffer.CourseDiffer_C1Type
    elif courseType.encode("utf-8") == 'C2类： 情景动画':
        result = CourseDiffer.CourseDiffer_C2TypeCartoon
    elif courseType.encode("utf-8") == 'C2类： 高端访谈':
        result = CourseDiffer.CourseDiffer_C2TypeHigh
    elif courseType.encode("utf-8") == 'EHS的VR实训课程':
        result = CourseDiffer.CourseDiffer_EHS
    elif courseType.encode("utf-8") == 'Excel进阶系列（Excel 2013）':
        result = CourseDiffer.CourseDiffer_Excel
    elif courseType.encode("utf-8") == 'TCP课程':
        result = CourseDiffer.CourseDiffer_TCP
    elif courseType.encode("utf-8") == 'UX用户体验':
        result = CourseDiffer.CourseDiffer_UXTest
    elif courseType.encode("utf-8") == '一带一路':
        result = CourseDiffer.CourseDiffer_YiDaiYiLu
    elif courseType.encode("utf-8") == '上海自贸区':
        result = CourseDiffer.CourseDiffer_ShangHaiZiMaoQu
    elif courseType.encode("utf-8") == '业务概况':
        result = CourseDiffer.CourseDiffer_YeWuGaiKuang
    elif courseType.encode("utf-8") == '个人业务':
        result = CourseDiffer.CourseDiffer_GeRenYeWu
    elif courseType.encode("utf-8") == '个人征信':
        result = CourseDiffer.CourseDiffer_GeRenZhengXin
    elif courseType.encode("utf-8") == '个人理财之案例说':
        result = CourseDiffer.CourseDiffer_GeRenLiCaiCase
    elif courseType.encode("utf-8") == '个人理财基础知识（QCBP）':
        result = CourseDiffer.CourseDiffer_GeRenLiCaiBase
    elif courseType.encode("utf-8") == '个人理财实务':
        result = CourseDiffer.CourseDiffer_GeRenLiCaiShiWu
    elif courseType.encode("utf-8") == '个人贷款 （QCBP）':
        result = CourseDiffer.CourseDiffer_GeRenDaiKuan
    elif courseType.encode("utf-8") == '个人风格与色彩搭配':
        result = CourseDiffer.CourseDiffer_GeRenStyleColor
    elif courseType.encode("utf-8") == '主题学习（通用）':
        result = CourseDiffer.CourseDiffer_TongyongTitleStudy
    elif courseType.encode("utf-8") == '云平台':
        result = CourseDiffer.CourseDiffer_YunPingTai
    elif courseType.encode("utf-8") == '互联网+':
        result = CourseDiffer.CourseDiffer_InternetPlus
    elif courseType.encode("utf-8") == '互联网环境下的生产与服务模式创新':
        result = CourseDiffer.CourseDiffer_InternetEnv
    elif courseType.encode("utf-8") == '产能提升':
        result = CourseDiffer.CourseDiffer_ImproveCapacity
    elif courseType.encode("utf-8") == '亲子阅读与情绪管理系列课程':
        result = CourseDiffer.CourseDiffer_ChildRead
    elif courseType.encode("utf-8") == '人力资源管理':
        result = CourseDiffer.CourseDiffer_ManageHumanResource
    elif courseType.encode("utf-8") == '企业文化':
        result = CourseDiffer.CourseDiffer_CompanyCulture
    elif courseType.encode("utf-8") == '企业知识萃取':
        result = CourseDiffer.CourseDiffer_CompanyKonwledge
    elif courseType.encode("utf-8") == '传世经典产品':
        result = CourseDiffer.CourseDiffer_ChuanShiJingDian
    elif courseType.encode("utf-8") == '保险一点通':
        result = CourseDiffer.CourseDiffer_InsureOnePoint
    elif courseType.encode("utf-8") == '保险基础理财':
        result = CourseDiffer.CourseDiffer_InsureBase
    elif courseType.encode("utf-8") == '保险基础知识':
        result = CourseDiffer.CourseDiffer_InsureBaseKnowledge
    elif courseType.encode("utf-8") == '保险的财富保全NDA':
        result = CourseDiffer.CourseDiffer_InsureNDA
    elif courseType.encode("utf-8") == '保险的财富管理功用':
        result = CourseDiffer.CourseDiffer_InsurePropertyManage
    elif courseType.encode("utf-8") == '保险科技':
        result = CourseDiffer.CourseDiffer_InsureKeJi
    elif courseType.encode("utf-8") == '保险销售':
        result = CourseDiffer.CourseDiffer_SellInsure
    elif courseType.encode("utf-8") == '保险顾问服务必做功课':
        result = CourseDiffer.CourseDiffer_InsureCounselorServer
    elif courseType.encode("utf-8") == '做合格党员':
        result = CourseDiffer.CourseDiffer_BeQualifiedPartyMember
    elif courseType.encode("utf-8") == '党建实务微课堂':
        result = CourseDiffer.CourseDiffer_MiniClassForDangJian
    elif courseType.encode("utf-8") == '党章知识选粹':
        result = CourseDiffer.CourseDiffer_DangZhangKnowledge
    elif courseType.encode("utf-8") == '入党第一课':
        result = CourseDiffer.CourseDiffer_FirstClassForJionParty
    elif courseType.encode("utf-8") == '其他':
        result = CourseDiffer.CourseDiffer_Other
    elif courseType.encode("utf-8") == '内训师（微课大赛&经验萃取）':
        result = CourseDiffer.CourseDiffer_NeiXunShi
    elif courseType.encode("utf-8") == '分类':
        result = CourseDiffer.CourseDiffer_Identify
    elif courseType.encode("utf-8") == '创新——来自方法而非灵感':
        result = CourseDiffer.CourseDiffer_InnovateFromFunc
    elif courseType.encode("utf-8") == '创新与创业':
        result = CourseDiffer.CourseDiffer_InnovateBeginBussiness
    elif courseType.encode("utf-8") == '创新企业战略与机会选择':
        result = CourseDiffer.CourseDiffer_InnovateWithCompany
    elif courseType.encode("utf-8") == '创新的方法':
        result = CourseDiffer.CourseDiffer_FuncOfInnovate
    elif courseType.encode("utf-8") == '区块链':
        result = CourseDiffer.CourseDiffer_QuKuaiLian
    elif courseType.encode("utf-8") == '博弈与决策':
        result = CourseDiffer.CourseDiffer_BoYiYuJueCe
    elif courseType.encode("utf-8") == '压力管理':
        result = CourseDiffer.CourseDiffer_YaLiManage
    elif courseType.encode("utf-8") == '参考报价':
        result = CourseDiffer.CourseDiffer_CanKaoBaoJia
    elif courseType.encode("utf-8") == '反洗钱基础':
        result = CourseDiffer.CourseDiffer_FanXiQianBase
    elif courseType.encode("utf-8") == '品牌战略':
        result = CourseDiffer.CourseDiffer_PingPaiZhanLue
    elif courseType.encode("utf-8") == '品鉴葡萄酒':
        result = CourseDiffer.CourseDiffer_TastingWine
    elif courseType.encode("utf-8") == '商业模式':
        result = CourseDiffer.CourseDiffer_BussinessModel
    elif courseType.encode("utf-8") == '商务礼仪':
        result = CourseDiffer.CourseDiffer_BussinesEtiquette
    elif courseType.encode("utf-8") == '太保人管系统操作':
        result = CourseDiffer.CourseDiffer_TaiBaoPersonManageSys
    elif courseType.encode("utf-8") == '央行反洗钱3号令解读':
        result = CourseDiffer.CourseDiffer_YangHangFanXiQian
    elif courseType.encode("utf-8") == '奢侈品系列':
        result = CourseDiffer.CourseDiffer_Luxury
    elif courseType.encode("utf-8") == '宏观经济':
        result = CourseDiffer.CourseDiffer_HongGuanJingJi
    elif courseType.encode("utf-8") == '客户服务':
        result = CourseDiffer.CourseDiffer_KeHuServer
    elif courseType.encode("utf-8") == '对冲基金':
        result = CourseDiffer.CourseDiffer_DuiChongJiJin
    elif courseType.encode("utf-8") == '小企业金融上岗培训':
        result = CourseDiffer.CourseDiffer_MiniCompanyJinRong
    elif courseType.encode("utf-8") == '市场营销':
        result = CourseDiffer.CourseDiffer_ShiChangYingXiao
    elif courseType.encode("utf-8") == '微课大赛':
        result = CourseDiffer.CourseDiffer_MiniClassMegagame
    elif courseType.encode("utf-8") == '微课大赛推送':
        result = CourseDiffer.CourseDiffer_MiniClassMegagameSend
    elif courseType.encode("utf-8") == '操作技能':
        result = CourseDiffer.CourseDiffer_OperateSkill
    elif courseType.encode("utf-8") == '教育心理学/系统化教学设计':
        result = CourseDiffer.CourseDiffer_EducatePsychology
    elif courseType.encode("utf-8") == '文化与伦理':
        result = CourseDiffer.CourseDiffer_CultureAndEthic
    elif courseType.encode("utf-8") == '新员工':
        result = CourseDiffer.CourseDiffer_NewEmployee
    elif courseType.encode("utf-8") == '新员工项目':
        result = CourseDiffer.CourseDiffer_NewEmployeeProject
    elif courseType.encode("utf-8") == '新国十条导论':
        result = CourseDiffer.CourseDiffer_NewCountryIntro
    elif courseType.encode("utf-8") == '新国十条详解':
        result = CourseDiffer.CourseDiffer_NewCountryDetail
    elif courseType.encode("utf-8") == '新媒体运营的意识、策划与实施':
        result = CourseDiffer.CourseDiffer_DoNewMedia
    elif courseType.encode("utf-8") == '日常运营':
        result = CourseDiffer.CourseDiffer_DailyRun
    elif courseType.encode("utf-8") == '易经系列课程':
        result = CourseDiffer.CourseDiffer_BookOfChangesClass
    elif courseType.encode("utf-8") == '晨夕会':
        result = CourseDiffer.CourseDiffer_MorningEveningMeet
    elif courseType.encode("utf-8") == '最佳实践萃取':
        result = CourseDiffer.CourseDiffer_BestPractice
    elif courseType.encode("utf-8") == '柜在知心':
        result = CourseDiffer.CourseDiffer_CounterInHeart
    elif courseType.encode("utf-8") == '案例教学':
        result = CourseDiffer.CourseDiffer_CaseTeach
    elif courseType.encode("utf-8") == '每个人都是VIP：需求为本销售技巧':
        result = CourseDiffer.CourseDiffer_EachPersonIsVIP
    elif courseType.encode("utf-8") == '每天都有好心情 ——兴趣，是一剂良药':
        result = CourseDiffer.CourseDiffer_GoodMoodIntrest
    elif courseType.encode("utf-8") == '每天都有好心情 ——成功感，八部曲':
        result = CourseDiffer.CourseDiffer_GoodMoodSuccess
    elif courseType.encode("utf-8") == '每天都有好心情——积极思维':
        result = CourseDiffer.CourseDiffer_GoodMoodPositiveThink
    elif courseType.encode("utf-8") == '法律宣传':
        result = CourseDiffer.CourseDiffer_LawPublicity
    elif courseType.encode("utf-8") == '活动运营（PK答题&闯关答题）':
        result = CourseDiffer.CourseDiffer_ActivityRunAnsower
    elif courseType.encode("utf-8") == '活动运营（直播）':
        result = CourseDiffer.CourseDiffer_ActivityRunLive
    elif courseType.encode("utf-8") == '活动运营（读书会）':
        result = CourseDiffer.CourseDiffer_ActivityRunReadBook
    elif courseType.encode("utf-8") == '玩转高尔夫':
        result = CourseDiffer.CourseDiffer_PlayGolf
    elif courseType.encode("utf-8") == '现场访谈/案例脚本':
        result = CourseDiffer.CourseDiffer_SceneInterview
    elif courseType.encode("utf-8") == '电子教材':
        result = CourseDiffer.CourseDiffer_ETextbook
    elif courseType.encode("utf-8") == '百科词条':
        result = CourseDiffer.CourseDiffer_CyclopediaEntry
    elif courseType.encode("utf-8") == '知识库建设':
        result = CourseDiffer.CourseDiffer_BuildKnowledgeLib
    elif courseType.encode("utf-8") == '知识手册':
        result = CourseDiffer.CourseDiffer_KnowledgeBrochure
    elif courseType.encode("utf-8") == '管理技能':
        result = CourseDiffer.CourseDiffer_ManageSkill
    elif courseType.encode("utf-8") == '管理理论':
        result = CourseDiffer.CourseDiffer_ManageTheory
    elif courseType.encode("utf-8") == '管理随身学':
        result = CourseDiffer.CourseDiffer_ManageStudyNearBody
    elif courseType.encode("utf-8") == '职业发展技能':
        result = CourseDiffer.CourseDiffer_ProfessionDevelopSkill
    elif courseType.encode("utf-8") == '职业道德':
        result = CourseDiffer.CourseDiffer_ProfessionMorality
    elif courseType.encode("utf-8") == '职场随身学':
        result = CourseDiffer.CourseDiffer_WorkplaceStudyNearBody
    elif courseType.encode("utf-8") == '自贸区':
        result = CourseDiffer.CourseDiffer_FreeTradeArea
    elif courseType.encode("utf-8") == '茶说新语':
        result = CourseDiffer.CourseDiffer_TeaTalkNewWord
    elif courseType.encode("utf-8") == '行政解读':
        result = CourseDiffer.CourseDiffer_AdministrationExplain
    elif courseType.encode("utf-8") == '视频设计':
        result = CourseDiffer.CourseDiffer_VedioDesign
    elif courseType.encode("utf-8") == '证券投资基金基础知识':
        result = CourseDiffer.CourseDiffer_SecurityInvest
    elif courseType.encode("utf-8") == '课程':
        result = CourseDiffer.CourseDiffer_Cource
    elif courseType.encode("utf-8") == '课程制作':
        result = CourseDiffer.CourseDiffer_MakeCourse
    elif courseType.encode("utf-8") == '课程设计与开发1.0':
        result = CourseDiffer.CourseDiffer_CourseDesignDevelopOne
    elif courseType.encode("utf-8") == '课程设计与开发2.0':
        result = CourseDiffer.CourseDiffer_CourseDesignDevelopTwo
    elif courseType.encode("utf-8") == '财富管理基础':
        result = CourseDiffer.CourseDiffer_BaseMeasureManage
    elif courseType.encode("utf-8") == '资格认证':
        result = CourseDiffer.CourseDiffer_QualificationAuthentication
    elif courseType.encode("utf-8") == '运营内控':
        result = CourseDiffer.CourseDiffer_RunInternalPiloting
    elif courseType.encode("utf-8") == '运营管理':
        result = CourseDiffer.CourseDiffer_RunManage
    elif courseType.encode("utf-8") == '通用管理学':
        result = CourseDiffer.CourseDiffer_TongYongManage
    elif courseType.encode("utf-8") == '部门展示':
        result = CourseDiffer.CourseDiffer_SectionShow
    elif courseType.encode("utf-8") == '金融改革':
        result = CourseDiffer.CourseDiffer_FinancialReform
    elif courseType.encode("utf-8") == '金融衍生品':
        result = CourseDiffer.CourseDiffer_FinancialDerivative
    elif courseType.encode("utf-8") == '银行业法律法规与综合能力(QCBP)':
        result = CourseDiffer.CourseDiffer_BankLawAndSkill
    elif courseType.encode("utf-8") == '银行服务':
        result = CourseDiffer.CourseDiffer_BankService
    elif courseType.encode("utf-8") == '银行柜面技巧':
        result = CourseDiffer.CourseDiffer_BankCounterAcrobatics
    elif courseType.encode("utf-8") == '银行柜面技能':
        result = CourseDiffer.CourseDiffer_BankCounterSkill
    elif courseType.encode("utf-8") == '银行管理（QCBP）':
        result = CourseDiffer.CourseDiffer_BankManage
    elif courseType.encode("utf-8") == '销售技能':
        result = CourseDiffer.CourseDiffer_SellSkill
    elif courseType.encode("utf-8") == '零售信贷上岗':
        result = CourseDiffer.CourseDiffer_RetailCreditWork
    elif courseType.encode("utf-8") == '面谈技能':
        result = CourseDiffer.CourseDiffer_InterviewSkill
    elif courseType.encode("utf-8") == '顶尖销售':
        result = CourseDiffer.CourseDiffer_TopSellor
    elif courseType.encode("utf-8") == '项目':
        result = CourseDiffer.CourseDiffer_Project
    elif courseType.encode("utf-8") == '领导力':
        result = CourseDiffer.CourseDiffer_Leadership
    elif courseType.encode("utf-8") == '风险合规':
        result = CourseDiffer.CourseDiffer_RiskCompliance
    elif courseType.encode("utf-8") == '高端销售典型案例分析':
        result = CourseDiffer.CourseDiffer_TopSellCase
    elif courseType.encode("utf-8") == '高管活动日宣传片':
        result = CourseDiffer.CourseDiffer_TopManager
    elif courseType.encode("utf-8") == '黄金投资':
        result = CourseDiffer.CourseDiffer_GoldInvest
    else:
        result = CourseDiffer.CourseDiffer_None

    return result


def makeFeature():
    logging.warning(u"运行日志：构建特征")
    user_feature = []
    course_feature = []

    user_dict, course_dict, user_data_dict, \
        course_data_dict = getAllUserAndCourse()

    user_length = len(user_dict)
    course_length = len(course_dict)
    max_point = 1
    # 没有积分数据，先注释掉
    # for key, value in user_data_dict.items():
        # if max_point < value[1]:
        #     max_point = value[1]

    for index in range(user_length):
        one_list = []
        # one_list.append(float(user_data_dict[index][1]) / max_point)
        #没有积分数据，临时改成1
        one_list.append(1)
        user_feature.append(one_list)

    for index in range(course_length):
        value = CourseType.CourseType_None
        if index in course_data_dict.keys():
            # value = transformCourseType(course_data_dict[index][1])
            #类别现在是id，现在不用transformCourseType转换
            value = course_data_dict[index][1]

        # course_feature.append(getEnumValue(value))
        # 类别现在是id，直接append即可
        course_feature.append(value)

    course_features = np.zeros((course_length, 136), dtype=np.float32)
    # for index in course_dict.keys():
    #     other_index = course_feature[index] - 1
    #     print other_index
    #     course_features[index][other_index] = 1

    #print(user_feature)
    #print('----------------')
    #print(course_features)

    return user_feature, course_features


def getAllUserAndCourse():
    logging.warning(u"运行日志：获取所有的用户和课程")
    u_nodes, v_nodes, ratings = [], [], []
    with open('gcn/toGcn.csv', 'r') as f:
    # with open('C:/Users/Administrator/Desktop/HybridRecommendGCN/gcn/toGcn.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u_nodes.append(int(row[0]))
            v_nodes.append(int(row[1]))
            ratings.append(int(row[2]))

    user_dict = {i: r for i, r in enumerate(list(set(u_nodes)))}
    course_dict = {i: r for i, r in enumerate(list(set(v_nodes)))}

    user_data_dict = getAllUserInfo(user_dict)
    course_data_dict = getAllCourseInfo(course_dict)

    return user_dict, course_dict, user_data_dict, course_data_dict


def getAllUserInfo(user_dict):
    logging.warning(u"运行日志：获取所有的用户信息")
    # sql = 'SELECT user_id, points, position, gender FROM user_basic_info'
    sql = """select id from account5000"""
    dbHandle = DatabaseIo()

    dataList = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)


    user_data_dict = makeDataDict(user_dict, dataList)

    return user_data_dict


def getAllCourseInfo(course_dict):
    logging.warning(u"运行日志：获取所有的课程信息")
    # sql = 'SELECT id, course_differ, course_type FROM course_info'
    sql = 'SELECT id, classify_id FROM course5000'

    dbHandle = DatabaseIo()

    dataList = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    course_data_dict = makeDataDict(course_dict, dataList)

    return course_data_dict


def makeDataDict(index_dict, data_list):
    # logging.warning(u"运行日志：构建数据字典")
    data_dict = {}
    for index in index_dict.keys():
        for one_data in data_list:
            if index_dict[index] == one_data[0]:
                data_dict[index] = one_data
                break

    return data_dict


if __name__ == '__main__':
    print("main")
    makeFeature()
