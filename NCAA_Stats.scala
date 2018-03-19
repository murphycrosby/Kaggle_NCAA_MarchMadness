// Databricks notebook source
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import sqlContext.implicits._

// COMMAND ----------

val ncaaresults = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/NCAATourneyDetailedResults.csv")
val regresults = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/RegularSeasonDetailedResults.csv")
val regresults_2018 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/RegularSeasonDetailedResults_2018.csv")

// COMMAND ----------

val games_1 = regresults.union(ncaaresults).union(regresults_2018)

// COMMAND ----------

val games_2 = (games_1.select($"Season"
                              ,$"DayNum"
                              ,$"WTeamID".alias("TeamID")
                              ,$"WScore".alias("Score")
                              ,$"WFGM".alias("FGM")
                              ,$"WFGA".alias("FGA")
                              ,$"WFGM3".alias("FGM3")
                              ,$"WFGA3".alias("FGA3")
                              ,$"WFTM".alias("FTM")
                              ,$"WFTA".alias("FTA")
                              ,$"WOR".alias("OR")
                              ,$"WDR".alias("DR")
                              ,$"WAst".alias("Ast")
                              ,$"WTO".alias("TO")
                              ,$"WStl".alias("Stl")
                              ,$"WBlk".alias("Blk")
                              ,$"WPF".alias("PF")
                             )
               .withColumn("Outcome",lit("W"))
               )

val games_3 = (games_1.select($"Season"
                              ,$"DayNum"
                              ,$"LTeamID".alias("TeamID")
                              ,$"LScore".alias("Score")
                              ,$"LFGM".alias("FGM")
                              ,$"LFGA".alias("FGA")
                              ,$"LFGM3".alias("FGM3")
                              ,$"LFGA3".alias("FGA3")
                              ,$"LFTM".alias("FTM")
                              ,$"LFTA".alias("FTA")
                              ,$"LOR".alias("OR")
                              ,$"LDR".alias("DR")
                              ,$"LAst".alias("Ast")
                              ,$"LTO".alias("TO")
                              ,$"LStl".alias("Stl")
                              ,$"LBlk".alias("Blk")
                              ,$"LPF".alias("PF")
                             )
               .withColumn("Outcome",lit("L"))
               )

val games = games_2.union(games_3).withColumn("Rank", rank().over(Window.partitionBy($"Season",$"TeamID").orderBy($"Season",$"DayNum")))

// COMMAND ----------

val g2 = games.select($"Season".alias("Season_2")
                      ,$"DayNum".alias("DayNum_2")
                      ,$"TeamID".alias("TeamID_2")
                      ,$"Score".alias("Score_2")
                      ,$"FGM".alias("FGM_2")
                      ,$"FGA".alias("FGA_2")
                      ,$"FGM3".alias("FGM3_2")
                      ,$"FGA3".alias("FGA3_2")
                      ,$"FTM".alias("FTM_2")
                      ,$"FTA".alias("FTA_2")
                      ,$"OR".alias("OR_2")
                      ,$"DR".alias("DR_2")
                      ,$"Ast".alias("Ast_2")
                      ,$"TO".alias("TO_2")
                      ,$"Stl".alias("Stl_2")
                      ,$"Blk".alias("Blk_2")
                      ,$"PF".alias("PF_2")
                      ,$"Rank".alias("Rank_2")
                      )

// COMMAND ----------

val games_join = (games.join(g2,games.col("Season") === g2.col("Season_2")
                       && games.col("TeamID") === g2.col("TeamID_2")
                       && games.col("Rank") > g2.col("Rank_2")
                       && (games.col("Rank")-3) <= g2.col("Rank_2")
                      ,"left")
            )

// COMMAND ----------

val stats = (games_join.groupBy($"Season",$"DayNum",$"TeamID",$"Outcome")
             .agg(avg("Score_2").alias("AvgScore")
                  ,(sum($"FGM_2")/sum($"FGA_2")).alias("AvgFG")
                  ,(sum($"FGM3_2")/sum($"FGA3_2")).alias("AvgFG3")
                  ,(sum($"FTM_2")/sum($"FTA_2")).alias("AvgFT")
                  ,avg("OR_2").alias("AvgOR")
                  ,avg("DR_2").alias("AvgDR")
                  ,avg("Ast_2").alias("AvgAst")
                  ,avg("TO_2").alias("AvgTO")
                  ,avg("Stl_2").alias("AvgStl")
                  ,avg("Blk_2").alias("AvgBlk")
                  ,avg("PF_2").alias("AvgPF")
                 )
             .orderBy($"Season",$"DayNum")
            )

// COMMAND ----------

stats.write.mode(SaveMode.Overwrite).format("orc").save("/FileStore/tables/NCAA_Stats/")

// COMMAND ----------

//display(games_1.filter("WTeamID=1328").select($"Season",$"WScore",$"LScore"))
