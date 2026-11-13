# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

products = spark.table("workspace.bronze.products").select(
    F.col("product_id").cast("int"),
    F.col("product_name"),
    F.col("aisle_id").cast("int"),
    F.col("department_id").cast("int"),
)

aisles = spark.table("workspace.bronze.aisles").select(
    F.col("aisle_id").cast("int"),
    F.col("aisle"),
)

departments = spark.table("workspace.bronze.departments").select(
    F.col("department_id").cast("int"),
    F.col("department"),
)

products_silver = (products
    .join(aisles, "aisle_id", "left")
    .join(departments, "department_id", "left")
    .dropDuplicates(["product_id"]))

(products_silver.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.silver.products"))

print(f"silver.products: {spark.table('workspace.silver.products').count():,}")

# COMMAND ----------

w_user = (Window.partitionBy("user_id")
          .orderBy("order_number")
          .rowsBetween(Window.unboundedPreceding, 0))

orders_silver = (spark.table("workspace.bronze.orders")
    .select(
        F.col("order_id").cast("int"),
        F.col("user_id").cast("int"),
        F.col("eval_set"),
        F.col("order_number").cast("int"),
        F.col("order_dow").cast("int"),
        F.col("order_hour_of_day").cast("int"),
        F.col("days_since_prior_order").cast("double"),
    )
    .filter(F.col("order_id").isNotNull() & F.col("user_id").isNotNull())
    .dropDuplicates(["order_id"])
    .withColumn(
        "dias_acumulados",
        F.sum(F.coalesce(F.col("days_since_prior_order"), F.lit(0.0))).over(w_user),
    )
    .withColumn(
        "mes_vida",
        (F.floor(F.col("dias_acumulados") / 30) + 1).cast("int"),
    ))

(orders_silver.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.silver.orders"))

print(f"silver.orders: {spark.table('workspace.silver.orders').count():,}")

# COMMAND ----------

op_union = (spark.table("workspace.bronze.order_products_prior")
                .withColumn("origen", F.lit("prior"))
            .unionByName(
             spark.table("workspace.bronze.order_products_train")
                .withColumn("origen", F.lit("train"))))

orders_keys = spark.table("workspace.silver.orders").select(
    "order_id", "user_id", "order_number")

products_keys = spark.table("workspace.silver.products").select(
    "product_id", "aisle", "department")

order_products_silver = (op_union
    .select(
        F.col("order_id").cast("int"),
        F.col("product_id").cast("int"),
        F.col("add_to_cart_order").cast("int"),
        F.col("reordered").cast("int"),
        F.col("origen"),
    )
    .dropDuplicates(["order_id", "product_id"])
    .join(orders_keys, "order_id", "inner")
    .join(products_keys, "product_id", "left"))

(order_products_silver.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.silver.order_products"))

print(f"silver.order_products: {spark.table('workspace.silver.order_products').count():,}")

# COMMAND ----------

o = spark.table("workspace.silver.orders")
opx = spark.table("workspace.silver.order_products")

print(f"clientes únicos:          {o.select('user_id').distinct().count():,}")
print(f"primeros pedidos (nulos): {o.filter(F.col('days_since_prior_order').isNull()).count():,}")
print(f"líneas sin departamento:  {opx.filter(F.col('department').isNull()).count():,}")
print(f"líneas sin cliente:       {opx.filter(F.col('user_id').isNull()).count():,}")

display(o.groupBy("mes_vida").count().orderBy("mes_vida"))

# COMMAND ----------

