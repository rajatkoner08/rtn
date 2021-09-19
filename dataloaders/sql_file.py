# Sebastian Raschka, 2014
# Creating a new SQLite database

import sqlite3

class Sql_ops():
        def __init__(self):
                sqlite_file = 'relations_db.sqlite'    # name of the sqlite database file
                # Connecting to the database file
                self.conn = sqlite3.connect(sqlite_file)
                self.c = self.conn.cursor()
                self.create_rels_master = 'CREATE TABLE if not exists pred_rels_master( srt_idx INTEGER, subject TEXT, object TEXT, perdicate TEXT, ' \
                                          'pred1 TEXT,pred2 TEXT,pred3 TEXT,pred4 TEXT, pred5 TEXT)'
                self.insert_rels = 'INSERT INTO pred_rels_master(srt_idx, subject, object, perdicate, pred1, pred2, pred3, pred4, pred5) VALUES(?,?,?,?,?,?,?,?,?)'
                # self.create_rels_master = 'CREATE TABLE if not exists rels_master( subject TEXT, perdicate TEXT, object TEXT, UNIQUE (subject, perdicate, object))'
                # self.insert_rels = 'INSERT INTO rels_master(subject, perdicate, object) VALUES(?,?,?)'

        def __del__(self):
                print('Commiting and Closing connection')
                self.conn.commit()
                self.conn.close()

        # def close_conn(self):
        #         self.conn.commit()
        #         self.conn.close()

        def create_table(self, query=None):
                try:
                        self.c.execute(query if query is not None else self.create_rels_master)
                        print('Table created')
                except sqlite3.Error as err:
                        print('Failed table creation err: ',err)

        def drop_table(self):
                try:
                        self.c.execute('drop table rels_master')
                except sqlite3.Error as err:
                        print(err)

        def insert_table(self, values, query=None):
                try:
                    self.c.execute(query if query is not None else self.insert_rels, values)
                except sqlite3.Error as error:
                    print('Cudnt insert : ', values,' and error msg ', error)
                finally:
                    self.conn.commit()

        def del_table(self, table_name='rels_master'):
                try:
                        self.c.execute('delete from '+table_name)
                        self.conn.commit()
                except sqlite3.Error as err:
                        print('Error in delete : ',err)

        def print_table(self, table_name):
                self.c.execute('SELECT * FROM ' +table_name)
                rows = self.c.fetchall()
                for row in rows:
                        print(row)

# if __name__ == "__main__":
#         print('calling sql ops')
#         a = Sql_ops()
        # drop_table()
        # create_table(create_rels_master)
        # insert_table(('man', 'hold', 'cup'),insert_rels)
        # insert_table(('flower', 'in', 'vase'),insert_rels)
        # insert_table(('man', 'hold', 'cup'),insert_rels)
        # print_table('rels_master')
