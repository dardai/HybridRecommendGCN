# -*- coding: utf-8 -*-

from globalConst import SqlType

SqlMap = {
    SqlType.GetRecommendClassNameByUid = """
             select course_name
             from course_info join course_dr ON course_info.id = course_dr.course_index
             where course_dr.user_id={}
    """,
    SqlType.GetRecommendClassScoreByCourseid = """
            select recommend_value
            from course_dr
            where course_index={}
    """,
}
