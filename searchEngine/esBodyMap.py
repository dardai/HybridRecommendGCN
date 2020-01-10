# -*- coding: utf-8 -*-

from globalConst import ESBodyType

ESBody = {
    ESBodyType.Create : {
        "mappings" : {
            "properties" : {
                "title" : {
                    "type" : "completion",
                },
                "courseid" : {
                    "type" : "long",
                }
            }
        }
    },
    ESBodyType.SearchOneWord : {
        "query" : {
            "match" : {
                "title" : None,
            }
        }
    },
    ESBodyType.SearchWords : {
        "query" : {
            "bool" : {
                "must" : []
            }
        }
    },
    ESBodyType.Suggester = {
        "suggest" : {
            "class-name-suggestion" : {
                "prefix" : "None",
                "completion" : {
                    "field" : "title"
                }
            }
        }
    },
    ESBodyType.SearchDocByCid = {
        "query" : {
            "match_phrase" : {
                'courseid' : courseId,
            }
        }
    }
    ESBodyType.SearchAll = {
        "query" : {
            "match_all" : {}
        }
    }
}
