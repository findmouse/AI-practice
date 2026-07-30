import json
import requests
import os
import pymysql
from dotenv import load_dotenv, find_dotenv

# todo: 1.データベースのテーブル構造を記述する（単一テーブル）
database_schema_string = """
  CREATE TABLE `emp` (
  `empno` int DEFAULT NULL, --社員番号, デフォルトはNULL
  `ename` varchar(50) DEFAULT NULL, --社員名, デフォルトはNULL
  `job` varchar(50) DEFAULT NULL,--職種, デフォルトはNULL
  `mgr` int DEFAULT NULL,--上司, デフォルトはNULL
  `hiredate` date DEFAULT NULL,--入社日, デフォルトはNULL
  `sal` int DEFAULT NULL,--月給, デフォルトはNULL
  `comm` int DEFAULT NULL,--賞与, デフォルトはNULL
  `deptno` int DEFAULT NULL,--部門番号, デフォルトはNULL
)"""

# todo: 2.データベースのテーブル構造を記述する（複数テーブル）
database_schema_string1 = """
CREATE TABLE `emp` (
  `empno` int DEFAULT NULL, --社員番号, デフォルトはNULL
  `ename` varchar(50) DEFAULT NULL, --社員名, デフォルトはNULL
  `job` varchar(50) DEFAULT NULL,--職種, デフォルトはNULL
  `mgr` int DEFAULT NULL,--上司, デフォルトはNULL
  `hiredate` date DEFAULT NULL,--入社日, デフォルトはNULL
  `sal` int DEFAULT NULL,--月給, デフォルトはNULL
  `comm` int DEFAULT NULL,--賞与, デフォルトはNULL
  `deptno` int DEFAULT NULL,--部門番号, デフォルトはNULL
);

CREATE TABLE `DEPT` (
  `DEPTNO` int NOT NULL, -- 部門コード, デフォルトはNULL
  `DNAME` varchar(14) DEFAULT NULL,--部門名, デフォルトはNULL
  `LOC` varchar(13) DEFAULT NULL,--所在地, デフォルトはNULL
  PRIMARY KEY (`DEPTNO`)
);

"""
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "この関数を使用して業務上の質問に回答してください。出力はSQLクエリ文である必要があります",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"ユーザーの質問に回答するための情報を抽出するSQLクエリ。"
                                       f"SQLは以下のデータベーススキーマを使用して記述してください:{database_schema_string1}"
                                       f"クエリはJSONではなく、プレーンテキストで返す必要があります。"
                                       f"クエリはMySQLがサポートする構文のみを含む必要があります。",
                    }
                },
                "required": ["query"],
            },
        }
    }
]


# todo:1.データベースに接続し、SQL文のクエリを実行する


def ask_database(query):
    """データベースに接続し、クエリを実行する"""
    # 1.MySQLデータベースに接続する
    print("関数内部に入りました")
    conn = pymysql.connect(
        host='localhost',
        port=13306,
        user='root',
        password='123456',
        database='itcast',
        charset='utf8mb4',  # カーソルクラスを指定し、結果を辞書型で返す
    )
    # 2. カーソルを作成する
    cursor = conn.cursor()
    print(f'テストを開始します')
    # 3. SQL文のテストを実行する
    # 例：SQLクエリを実行する
    # sql = "SELECT * FROM emp"
    print(f'query--》{query}')
    cursor.execute(query)
    # 4. クエリ結果を取得する
    result = cursor.fetchall()
    # 5.カーソルを閉じる
    cursor.close()
    # 6.接続を閉じる
    conn.close()
    return result


# # todo: 2.モデルの返答に応じて使用するツール関数を判断する：
def parse_response(response):
    response_message = response.choices[0].message
    # 関数呼び出しが必要かどうかを判定する
    if response_message.tool_calls:
        # 関数を呼び出す
        available_functions = {
            "ask_database": ask_database
        }  # only one function test in this example, but you can have multiple
        function_name = response_message.tool_calls[0].function.name
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        function_response = fuction_to_call(
            query=function_args.get("query"),
        )
        return function_response


if __name__ == '__main__':
    query = "select count(*) from emp"
    a = ask_database(query)
    print(a)
