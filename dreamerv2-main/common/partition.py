import copy
import csv
import gym
import numpy as np

class Environment(gym.Env):

    #Initializer. Calls the reset method
    def __init__(self):

      #Step 1: We define all basic variables
      self.n_columns = 27
      self.n_actions =  self.n_columns + 1#Action 0 will be exit
      self.n_queries = 20
      self.action_space = gym.spaces.Discrete(self.n_actions)
      self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_queries,self.n_columns, 1), dtype=np.float32)

      #Step 2: We define the mapping of action number to partitioning column.

      self.schema = {#Also filtering those columns not used as predicates in the TPC-H queries.
         "nation": sorted(set(["n_nationkey", "n_name", "n_regionkey"])),#"n_comment"
         "region": sorted(set(["r_regionkey", "r_name"])),#"r_comment"
         "part": sorted(set(["p_brand", "p_type", "p_size", "p_container"])),#"p_partkey", "p_comment", "p_name", "p_mfgr", "p_retailprice"
         "supplier": sorted(set(["s_nationkey"])),#"s_suppkey", "s_address", "s_comment", "s_name", "s_phone", "s_acctbal"
         "partsupp": sorted(set(["ps_availqty", "ps_supplycost"])),#"ps_partkey", "ps_suppkey", "ps_comment"
         "customer": sorted(set(["c_nationkey", "c_phone", "c_acctbal", "c_mktsegment"])),#"c_custkey", "c_name", "c_address", "c_comment"
         "orders": sorted(set(["o_orderstatus", "o_orderdate", "o_orderpriority"])),#"o_orderkey". "o_custkey", "o_clerk", "o_comment", "o_totalprice", "o_shippriority"
         "lineitem": sorted(set(["l_quantity", "l_discount", "l_returnflag", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode"]))#"l_orderkey", "l_partkey",         "l_suppkey", "l_comment", "l_linenumber", "l_extendedprice", "l_tax", "l_linestatus"
      }

      self.sorted_columns = []
      self.columns_to_tables = {}
      for table in sorted(self.schema.keys()):
        for col in self.schema[table]:
          self.columns_to_tables[col]=table
          self.sorted_columns.append(col)
      #To find the action number given a column we do: action = 1 + self.sorted_columns.index(col) 
      #To find the column given an action we do: self.sorted_columns[action-1]<=This makes sense for all actions except 0

      #Step 3: We load the runtimes
      self.partition_to_runtime = {}
      with open('common/TPCH_SF10_runtimes_partitions.csv') as csv_file:
          csv_reader = csv.reader(csv_file)
          line_count = 0
          for row in csv_reader:
            if line_count != 0:
              self.partition_to_runtime[row[0]]=float(row[1])
            line_count+=1
      self.default_runtime = self.partition_to_runtime[""]

      #Step 4: We define the queries, which we will use afterwards to define and update the state
      self.q1 = "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= date '1998-12-01' - INTERVAL '90' day group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus"
      self.q2 = """select
          s_acctbal,
          s_name,
          n_name,
          p_partkey,
          p_mfgr,
          s_address,
          s_phone,
          s_comment
        from
          part,
          supplier,
          partsupp,
          nation,
          region
        where
          p_partkey = ps_partkey
          and s_suppkey = ps_suppkey
          and p_size = 15
          and p_type like '%BRASS'
          and s_nationkey = n_nationkey
          and n_regionkey = r_regionkey
          and r_name = 'EUROPE'
          and ps_supplycost = (
            select
              min(ps_supplycost)
            from
              partsupp,
              supplier,
              nation,
              region
            where
              p_partkey = ps_partkey
              and s_suppkey = ps_suppkey
              and s_nationkey = n_nationkey
              and n_regionkey = r_regionkey
              and r_name = 'EUROPE'
          )
        order by
          s_acctbal desc,
          n_name,
          s_name,
          p_partkey
        """

      self.q3 = """select
          l_orderkey,
          sum(l_extendedprice * (1 - l_discount)) as revenue,
          o_orderdate,
          o_shippriority
        from
          customer,
          orders,
          lineitem
        where
          c_mktsegment = 'BUILDING'
          and c_custkey = o_custkey
          and l_orderkey = o_orderkey
          and o_orderdate < date '1995-03-15'
          and l_shipdate > date '1995-03-15'
        group by
          l_orderkey,
          o_orderdate,
          o_shippriority
        order by
          revenue desc,
          o_orderdate
        """

      self.q4 = """select
          o_orderpriority,
          count(*) as order_count
        from
          orders
        where
          o_orderdate >= date '1993-07-01'
          and o_orderdate < date '1993-07-01' + interval '3' month
          and exists (
            select
              *
            from
              lineitem
            where
              l_orderkey = o_orderkey
              and l_commitdate < l_receiptdate
          )
        group by
          o_orderpriority
        order by
          o_orderpriority
        """

      self.q5 = """select
          n_name,
          sum(l_extendedprice * (1 - l_discount)) as revenue
        from
          customer,
          orders,
          lineitem,
          supplier,
          nation,
          region
        where
          c_custkey = o_custkey
          and l_orderkey = o_orderkey
          and l_suppkey = s_suppkey
          and c_nationkey = s_nationkey
          and s_nationkey = n_nationkey
          and n_regionkey = r_regionkey
          and r_name = 'ASIA'
          and o_orderdate >= date '1994-01-01'
          and o_orderdate < date '1994-01-01' + interval '1' year
        group by
          n_name
        order by
          revenue desc
        """

      self.q6 = """select
          sum(l_extendedprice * l_discount) as revenue
        from
          lineitem
        where
          l_shipdate >= date '1994-01-01'
          and l_shipdate < date '1994-01-01' + interval '1' year
          and l_discount between .06 - 0.01 and .06 + 0.01
          and l_quantity < 24
        """

      self.q7 = """select
          supp_nation,
          cust_nation,
          l_year,
          sum(volume) as revenue
        from
          (
            select
              n1.n_name as supp_nation,
              n2.n_name as cust_nation,
              extract(year from l_shipdate) as l_year,
              l_extendedprice * (1 - l_discount) as volume
            from
              supplier,
              lineitem,
              orders,
              customer,
              nation n1,
              nation n2
            where
              s_suppkey = l_suppkey
              and o_orderkey = l_orderkey
              and c_custkey = o_custkey
              and s_nationkey = n1.n_nationkey
              and c_nationkey = n2.n_nationkey
              and (
                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
              )
              and l_shipdate between date '1995-01-01' and date '1996-12-31'
          ) as shipping
        group by
          supp_nation,
          cust_nation,
          l_year
        order by
          supp_nation,
          cust_nation,
          l_year
        """

      self.q8 = """select
          o_year,
          sum(case
            when nation = 'BRAZIL' then volume
            else 0
          end) / sum(volume) as mkt_share
        from
          (
            select
              extract(year from o_orderdate) as o_year,
              l_extendedprice * (1 - l_discount) as volume,
              n2.n_name as nation
            from
              part,
              supplier,
              lineitem,
              orders,
              customer,
              nation n1,
              nation n2,
              region
            where
              p_partkey = l_partkey
              and s_suppkey = l_suppkey
              and l_orderkey = o_orderkey
              and o_custkey = c_custkey
              and c_nationkey = n1.n_nationkey
              and n1.n_regionkey = r_regionkey
              and r_name = 'AMERICA'
              and s_nationkey = n2.n_nationkey
              and o_orderdate between date '1995-01-01' and date '1996-12-31'
              and p_type = 'ECONOMY ANODIZED STEEL'
          ) as all_nations
        group by
          o_year
        order by
          o_year
        """

      self.q9 = """select
          nation,
          o_year,
          sum(amount) as sum_profit
        from
          (
            select
              n_name as nation,
              extract(year from o_orderdate) as o_year,
              l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
            from
              part,
              supplier,
              lineitem,
              partsupp,
              orders,
              nation
            where
              s_suppkey = l_suppkey
              and ps_suppkey = l_suppkey
              and ps_partkey = l_partkey
              and p_partkey = l_partkey
              and o_orderkey = l_orderkey
              and s_nationkey = n_nationkey
              and p_name like '%green%'
          ) as profit
        group by
          nation,
          o_year
        order by
          nation,
          o_year desc
        """

      self.q10 = """select
          c_custkey,
          c_name,
          sum(l_extendedprice * (1 - l_discount)) as revenue,
          c_acctbal,
          n_name,
          c_address,
          c_phone,
          c_comment
        from
          customer,
          orders,
          lineitem,
          nation
        where
          c_custkey = o_custkey
          and l_orderkey = o_orderkey
          and o_orderdate >= date '1993-10-01'
          and o_orderdate < date '1993-10-01' + interval '3' month
          and l_returnflag = 'R'
          and c_nationkey = n_nationkey
        group by
          c_custkey,
          c_name,
          c_acctbal,
          c_phone,
          n_name,
          c_address,
          c_comment
        order by
          revenue desc
        """

      self.q11 = """select
          ps_partkey,
          sum(ps_supplycost * ps_availqty) as value
        from
          partsupp,
          supplier,
          nation
        where
          ps_suppkey = s_suppkey
          and s_nationkey = n_nationkey
          and n_name = 'GERMANY'
        group by
          ps_partkey having
            sum(ps_supplycost * ps_availqty) > (
              select
                sum(ps_supplycost * ps_availqty) * 0.0001000000
              from
                partsupp,
                supplier,
                nation
              where
                ps_suppkey = s_suppkey
                and s_nationkey = n_nationkey
                and n_name = 'GERMANY'
            )
        order by
          value desc
        """

      self.q12 = """select
          l_shipmode,
          sum(case
            when o_orderpriority = '1-URGENT'
              or o_orderpriority = '2-HIGH'
              then 1
            else 0
          end) as high_line_count,
          sum(case
            when o_orderpriority <> '1-URGENT'
              and o_orderpriority <> '2-HIGH'
              then 1
            else 0
          end) as low_line_count
        from
          orders,
          lineitem
        where
          o_orderkey = l_orderkey
          and l_shipmode in ('MAIL', 'SHIP')
          and l_commitdate < l_receiptdate
          and l_shipdate < l_commitdate
          and l_receiptdate >= date '1994-01-01'
          and l_receiptdate < date '1994-01-01' + interval '1' year
        group by
          l_shipmode
        order by
          l_shipmode
        """

      self.q13 = """select
          c_count,
          count(*) as custdist
        from
          (
            select
              c_custkey,
              count(o_orderkey)
            from
              customer left outer join orders on
                c_custkey = o_custkey
                and o_comment not like '%special%requests%'
            group by
              c_custkey
          ) as c_orders (c_custkey, c_count)
        group by
          c_count
        order by
          custdist desc,
          c_count desc
        """

      self.q14 = """select
          100.00 * sum(case
            when p_type like 'PROMO%'
              then l_extendedprice * (1 - l_discount)
            else 0
          end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
        from
          lineitem,
          part
        where
          l_partkey = p_partkey
          and l_shipdate >= date '1995-09-01'
          and l_shipdate < date '1995-09-01' + interval '1' month
        """

      self.q16 = """select
          p_brand,
          p_type,
          p_size,
          count(distinct ps_suppkey) as supplier_cnt
        from
          partsupp,
          part
        where
          p_partkey = ps_partkey
          and p_brand <> 'Brand#45'
          and p_type not like 'MEDIUM POLISHED%'
          and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
          and ps_suppkey not in (
            select
              s_suppkey
            from
              supplier
            where
              s_comment like '%Customer%Complaints%'
          )
        group by
          p_brand,
          p_type,
          p_size
        order by
          supplier_cnt desc,
          p_brand,
          p_type,
          p_size
        """

      self.q17 = """select
          sum(l_extendedprice) / 7.0 as avg_yearly
        from
          lineitem,
          part
        where
          p_partkey = l_partkey
          and p_brand = 'Brand#23'
          and p_container = 'MED BOX'
          and l_quantity < (
            select
              0.2 * avg(l_quantity)
            from
              lineitem
            where
              l_partkey = p_partkey
          )
        """

      self.q18 = """select
          c_name,
          c_custkey,
          o_orderkey,
          o_orderdate,
          o_totalprice,
          sum(l_quantity)
        from
          customer,
          orders,
          lineitem
        where
          o_orderkey in (
            select
              l_orderkey
            from
              lineitem
            group by
              l_orderkey having
                sum(l_quantity) > 300
          )
          and c_custkey = o_custkey
          and o_orderkey = l_orderkey
        group by
          c_name,
          c_custkey,
          o_orderkey,
          o_orderdate,
          o_totalprice
        order by
          o_totalprice desc,
          o_orderdate
        """

      self.q19 = """select
          sum(l_extendedprice* (1 - l_discount)) as revenue
        from
          lineitem,
          part
        where
          (
            p_partkey = l_partkey
            and p_brand = 'Brand#12'
            and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
            and l_quantity >= 1 and l_quantity <= 1 + 10
            and p_size between 1 and 5
            and l_shipmode in ('AIR', 'AIR REG')
            and l_shipinstruct = 'DELIVER IN PERSON'
          )
          or
          (
            p_partkey = l_partkey
            and p_brand = 'Brand#23'
            and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
            and l_quantity >= 10 and l_quantity <= 10 + 10
            and p_size between 1 and 10
            and l_shipmode in ('AIR', 'AIR REG')
            and l_shipinstruct = 'DELIVER IN PERSON'
          )
          or
          (
            p_partkey = l_partkey
            and p_brand = 'Brand#34'
            and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
            and l_quantity >= 20 and l_quantity <= 20 + 10
            and p_size between 1 and 15
            and l_shipmode in ('AIR', 'AIR REG')
            and l_shipinstruct = 'DELIVER IN PERSON'
          )
        """

      self.q20 = """select
          s_name,
          s_address
        from
          supplier,
          nation
        where
          s_suppkey in (
            select
              ps_suppkey
            from
              partsupp
            where
              ps_partkey in (
                select
                  p_partkey
                from
                  part
                where
                  p_name like 'forest%'
              )
              and ps_availqty > (
                select
                  0.5 * sum(l_quantity)
                from
                  lineitem
                where
                  l_partkey = ps_partkey
                  and l_suppkey = ps_suppkey
                  and l_shipdate >= date '1994-01-01'
                  and l_shipdate < date '1994-01-01' + interval '1' year
              )
          )
          and s_nationkey = n_nationkey
          and n_name = 'CANADA'
        order by
          s_name
        """

      self.q21 = """select
          s_name,
          count(*) as numwait
        from
          supplier,
          lineitem l1,
          orders,
          nation
        where
          s_suppkey = l1.l_suppkey
          and o_orderkey = l1.l_orderkey
          and o_orderstatus = 'F'
          and l1.l_receiptdate > l1.l_commitdate
          and exists (
            select
              *
            from
              lineitem l2
            where
              l2.l_orderkey = l1.l_orderkey
              and l2.l_suppkey <> l1.l_suppkey
          )
          and not exists (
            select
              *
            from
              lineitem l3
            where
              l3.l_orderkey = l1.l_orderkey
              and l3.l_suppkey <> l1.l_suppkey
              and l3.l_receiptdate > l3.l_commitdate
          )
          and s_nationkey = n_nationkey
          and n_name = 'SAUDI ARABIA'
        group by
          s_name
        order by
          numwait desc,
          s_name
        """

      self.q22 = """select
          cntrycode,
          count(*) as numcust,
          sum(c_acctbal) as totacctbal
        from
          (
            select
              substring(c_phone from 1 for 2) as cntrycode,
              c_acctbal
            from
              customer_parquet
            where
              substring(c_phone from 1 for 2) in
                ('13', '31', '23', '29', '30', '18', '17')
              and c_acctbal > (
                select
                  avg(c_acctbal)
                from
                  customer_parquet
                where
                  c_acctbal > 0.00
                  and substring(c_phone from 1 for 2) in
                    ('13', '31', '23', '29', '30', '18', '17')
              )
              and not exists (
                select
                  *
                from
                  orders_parquet
                where
                  o_custkey = c_custkey
              )
          ) as custsale
        group by
          cntrycode
        order by
          cntrycode
        """
      self.queries=[self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7, self.q8, self.q9, self.q10, self.q11, self.q12, self.q13, self.q14, self.q16, self.q17, self.q18, self.q19, self.q20, self.q21]
      self.reset()
      

    #The reset function initializes the state, current partitions and action mask
    def reset(self):
      self.step_counter = 0
      self.current_partitions = set()
      self._update_state()
      self.action_mask = np.ones((self.n_actions), dtype=np.int32)
      return copy.deepcopy(np.expand_dims(self.state, axis=-1))

    def _update_state(self):
      self.state = np.zeros((self.n_queries,self.n_columns), dtype=np.float32)
      q_counter = 0
      for query in self.queries:
        for table in sorted(self.schema.keys()):
          table_used = False
          cols_used = set()
          for col in self.schema[table]:
            if col in query:
              table_used = True
              cols_used.add(col)
          if table_used:
            partition_on_table = None
            for part in self.current_partitions:
              if self.columns_to_tables[part]==table:
                partition_on_table = part
            if partition_on_table == None or (partition_on_table not in cols_used):
              for col in self.schema[table]:#If we need to read one column we read all
                self.state[q_counter][self.sorted_columns.index(col)]=1.0
            else:
              for col in cols_used:#If we are partitioned on one column, we read only from it (but for the others it is not too clear... for simplicity I'll just add them reading and leave other columns out)
                self.state[q_counter][self.sorted_columns.index(col)]=1.0
              
        q_counter+=1

    def close(self,):
      return None 

    @property
    def _n_actions(self):
      return self.n_actions

    def _get_obs(self):
      return copy.deepcopy(np.expand_dims(self.state, axis=-1))
    
    def _get_action_mask(self):
      self.action_mask = np.ones((self.n_actions), dtype=np.int32)
      for action in self.current_partitions:
        for col in self.schema[self.columns_to_tables[action]]:
          self.action_mask[1 + self.sorted_columns.index(col)] = 0
      return copy.deepcopy(self.action_mask)
    
    def _get_reward(self):
      key = "-".join(sorted(self.current_partitions))
      val = 0
      if key in self.partition_to_runtime:
        val = self.partition_to_runtime[key]
      else:
        val = 361699
      return max((10*(self.default_runtime/val)-1),0.)

    def step(self, action):
      if action==0:
        return copy.deepcopy(np.expand_dims(self.state, axis=-1)), self._get_reward(), True, {}     
      self.step_counter+=1
      if action<0 or action>=len(self.action_mask) or self._get_action_mask()[action]==0.:
        return copy.deepcopy(np.expand_dims(self.state, axis=-1)), -1., self.step_counter>=3, {}
      #Now we apply the action...
      current_partition = self.sorted_columns[action-1]
      self.current_partitions.add(current_partition)      
      self._update_state()
      if self.step_counter>=3:
          return copy.deepcopy(np.expand_dims(self.state, axis=-1)), self._get_reward(), self.step_counter>=3, {}
      else:
          return copy.deepcopy(np.expand_dims(self.state, axis=-1)), 0., self.step_counter>=3, {}
