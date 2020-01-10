# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch

from globalConst import ESBodyType, ESOperateErrorCode
from utils.extends import printSet
from searchEngine.esBodyMap import ESBody

'''
    在 Elasticsearch 中将不同的模块封装成了不同的类
    其中主要有一下的类:Transport, TransportError,
    IndicesClient, IngestClient, ClusterClient, CatClient,
    NodesClient, RemoteClient, SnapshotClient, TasksClient,
    XPackClient

    最基础开发使用 Elasticsearch类 和 Elasticsearch.indices类 中的函数就可以满足
    Elasticsearch.xxxx() 是一些对 Elasticsearch 的基础操作
    Elasticsearch.indices.xxxx() 是专门对 索引 进行操作的操作集
'''

'''
    用于判读部分操作是否成功
    操作包括: elasticsearch.index, elasticsearch.search
'''

'''
    Elasticsearch 文档结构
    id              序号        # 针对海量数据, 且为了避免重复, es中为了解决并发冲突, 自动生成的id可以避免这个问题, 而在查找时可以通过courseId来查找
    courseName      课程名
    author          作者
    courseid        课程ID      # 用与搜索出来结果查找对应的课程
'''

def isESOperateSuccess(result):
    if result and isinstance(result, dict) and result['_share'] and result['_share'].failed and result['_share'].failed == 0:
        return True
    return False

class elasticsearchPort():

    def __init__(self, host = None, indexName = None):
        self.m_host = host
        self.m_index = indexName
        self.m_es = Elasticsearch(host = host)
        self.m_have_index = False
        if indexName and isinstance(indexName, str) and len(indexName) > 0:
            self.m_have_index = True

    '''
    创建索引函数原型
    Elasticsearch.create(index, id, body, doc_type="_doc", params=None)
    Elasticsearch.indices.create(index, body=None, params=None)
    arg : body 中传递 索引 setting 和 mapping;
            setting 是 索引 的设置, 有分片等信息;
            mapping 是 索引 的表项设置
    arg : params 中传递查询的条件信息
    '''
    def createIndex(self):
        if self.m_es.indices.exists(self.m_index):
            return getEnumValue(ESOperateErrorCode.IndexExist)

        body = ESBody[ESBodyType.Create]
        self.m_es.indices.create(self.m_index, body, params, ignore=400)

        return getEnumValue(ESOperateErrorCode.Success)

    '''
        set函数, 设置indexName
    '''
    def setIndexName(self, indexName):
        if not isinstance(indexName, str):
            return getEnumValue(ESOperateErrorCode.ParamTypeError)

        self.m_index = indexName
        if len(self.m_index) > 0:
            self.m_have_index = True
            return getEnumValue(ESOperateErrorCode.Success)
        else:
            return getEnumValue(ESOperateErrorCode.NoneIndex)

    # 检测索引是否存在
    def isIndexExist(self, params = None):
        if not self.m_have_index:
            return getEnumValue(ESOperateErrorCode.ParamError)

        if self.m_es.indices.exists(self.m_index, params = params):
            return True
        return False

    '''
    用于初始化插入所有的数据
    函数原型: Elasticsearch.index(self, index, body, doc_type="_doc", id=None, params=None)
    index() 可以用于插入数据, 也可以进行更新
    '''
    def insertAllDate(self, dataTuple):
        if not self.m_have_index:
            return getEnumValue(ESOperateErrorCode.ParamError)
        if not isinstance(dataTuple, tuple):
            return getEnumValue(ESOperateErrorCode.ParamTypeError)
        if len(dataTuple) == 0:
            return getEnumValue(ESOperateErrorCode.ParamTypeError)

        insertErrorList = list()

        for one_item in dataTuple:
            if not isinstance(one_item, dict):
                insertErrorList.append(one_item)
                continue

            result = self.m_es.index(index = self.m_index, body = one_item)
            if isESOperateSuccess(result) > 0:      # index是一条一条插入, 判断 failed 应该就够了
                insertErrorList.append(one_item)
                continue

        if len(insertErrorList) > 0:
            return False, insertErrorList
        return True, insertErrorList

    def searchByContext(self, context, params = None):
        body
        if isinstance(context, str):
            body = ESBody[ESBodyType.SearchOneWord]
            body['query']['match']['title'] = context
        if isinstance(context, list):
            body = ESBody[ESBodyType.SearchWords]
            for key_word in context:
                body['query']['bool']['must'].append({ "match" : { "title" : key_word } })

        result = self.m_es.search(index = self.m_index, body = body, params = params)

        return result

    def suggesterByContext(self, context):
        body = ESBody[ESBodyType.Suggester]
        body["suggest"]["class-name-suggestion"]["prefix"] = context

        result = self.m_es.search(index = self.m_index, body = body)

        return result

    # 单项插入
    def insertOneDate(self, data):
        if not self.m_have_index:
            return getEnumValue(ESOperateErrorCode.ParamError)
        if not isinstance(data, dict):
            return getEnumValue(ESOperateErrorCode.ParamTypeError)

        if not self.m_es.exists(self.m_index, data['courseid']):

            result = self.m_es.index(index = self.m_index, body = self.one_item)

            if isESOperateSuccess(result) > 0:      # index是一条一条插入, 判断 failed 应该就够了
                return getEnumValue(ESOperateErrorCode.InsertError)
            else:
                return getEnumValue(ESOperateErrorCode.Success)
        else:
            return getEnumValue(ESOperateErrorCode.DocIdExist)

    def updateDateByCourseid(self, data):
        if not isinstance(data, dict):
            return getEnumValue(ESOperateErrorCode.ParamTypeError)

        if not self.m_es.exists(self.m_index, data['courseid']):
            return getEnumValue(ESOperateErrorCode.DocIdNotExist)

        docid = self.getDocIdByCourseid(data['courseid'])

        result = self.m_es.update(self.m_index, docid, doc_type="_doc", body=None, params=None)

        if isESOperateSuccess(result):
            return getEnumValue(ESOperateErrorCode.Success)
        return getEnumValue(ESOperateErrorCode.Faild)

    def getDocIdByCourseid(self, courseId):
        if not isinstance(courseId, int):
            return getEnumValue(ESOperateErrorCode.ParamError)

        body = ESBody[ESBodyType.SearchDocByCid]
        result = self.m_es.search(index = self.m_index, body = body)

        if len(result['_share']['hits']['hits']) > 0:
            for one_item in result['_share']['hits']['hits']:
                if one_item['courseid'] == courseId:
                    return one_item['id']
        else:
            return 0


    def getAllData(self):
        if not self.m_have_index:
            return getEnumValue(ESOperateErrorCode.ParamError)

        body = ESBody[ESBodyType.SearchAll]
        result = self.m_es.search(index = self.m_index, body = body)

        if isESOperateSuccess(result):
            return getEnumValue(ESOperateErrorCode.Success)
        return getEnumValue(ESOperateErrorCode.Faild)


    '''
    es.search() 中主要通过 body 来传递你要查询的条件
    body = {
        "query" :
        {
            "match" :           # style 是字段名, 后面的匹配的内容, 将两个词分别匹配
            {
                'style' : '冒险 战斗',
            },
            "match_phrase" :    # 在 name 中匹配整个短语, 完全匹配
            {
                'name' : 'fate stay night',
            }
            "bool" :
            {
                "must" :        # 即( && )
                [
                    { "match" : { "country" : "日本" } },
                    { "match" : { "country" : "美国" } }
                ],
                "should" :      # 即( || )
                [
                    { "match" : { "country" : "日本" } },
                    { "match" : { "country" : "美国" } }
                ],
                "must_not" :    # 即 !( && )
                [
                    { "match" : { "country" : "日本" } },
                    { "match" : { "date" : "2012" } }
                ],
                "filter" :
                [
                    { "range" : { "data" : { "gt" : 2010, "lt" : 2018 } } },    # gt : > ; gte : >= ; lt : < ; lte : <= ;
                ]
            }
        }
    }
    # 对于 params 这个参数, 传递进去的是 k, v 均为 str 的字典, 在连接之前将会将键值对和 url 进行拼接
    # 解析在 search 函数中可以使用的参数
    search params = {
        "_source"           : "true",       # 这个参数用于判断是否返回 _source 字段 可选的其他参数: false
        "_source_exclude"   : ""            # 这个参数好像没有, 会报错
        "_source_excludes"  : ""            # 这个参数用于包含部分 _source 中的属性, 其中可以添加多个属性名, 属性名之间用 ',' 隔开, 切中间不能有空格
        "_source_include"   : ""            # 这个参数好像没有, 会报错
        "_source_includes"  : ""            # 这个参数用于包含部分 _source 中的属性, 其中可以添加多个属性名, 属性名之间用 ',' 隔开, 切中间不能有空格
        "allow_no_indices"  : "true",       # 这个参数用于 index 中通配符的情况, 如果是 false, 将无法进行适配, 这个默认是开启的
        "analyze_wildcard"  : ""            # 这个参数好像没有, 会报错
        "analyzer"          : ""            # 这个参数好像没有, 会报错
        "conflicts"         : "proceed"     # 这个参数好像没有, 会报错, 常用于在更新语句中, 出现冲突时, 确定是否返回 可选的其他参数: true
    }
    '''




