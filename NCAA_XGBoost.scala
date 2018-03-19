// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.{XGBoost}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import sqlContext.implicits._

// COMMAND ----------

val coaches = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/TeamCoaches.csv")
val teams = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/Teams.csv")
val cities = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/Cities.csv")
val game_cities = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/GameCities.csv")
val ncaaresults = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/NCAATourneyDetailedResults.csv")
val regresults = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/RegularSeasonDetailedResults.csv")
val regresults_2018 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/RegularSeasonDetailedResults_2018.csv")
val stats = spark.read.format("orc").load("/FileStore/tables/NCAA_Stats/")
val elo = spark.read.format("orc").load("/FileStore/tables/elo/")

// COMMAND ----------

val games_1 = regresults.union(ncaaresults).union(regresults_2018)

// COMMAND ----------

val games_2 = (games_1.join(game_cities, games_1.col("Season") === game_cities.col("Season") 
                            && games_1.col("DayNum") === game_cities.col("DayNum")
                            && games_1.col("WTeamID") === game_cities.col("WTeamID")
                            && games_1.col("LTeamID") === game_cities.col("LTeamID")
                            ,"left")
               .drop(game_cities.col("Season"))
               .drop(game_cities.col("DayNum"))
               .drop(game_cities.col("WTeamID"))
               .drop(game_cities.col("LTeamID"))
              )

// COMMAND ----------

/*
val games_2_5 = (games_2.withColumnRenamed("CRType","CRType_1")
                 .withColumnRenamed("CityID","CityID_1")
                 .withColumn("CRType",when($"CRType_1".isNull,lit("Unknown")).otherwise($"CRType_1"))
                 .withColumn("CityID",when($"CityID_1".isNull,lit(0)).otherwise($"CityID_1"))
                 .drop($"CRType_1")
                 .drop($"CityID_1")
             )
*/
val games_2_5 = (games_2.withColumnRenamed("CRType","CRType_1")
                 .withColumn("CRType",when($"CRType_1".isNull,lit("Unknown")).otherwise($"CRType_1"))
                 .drop($"CRType_1")
             )

// COMMAND ----------

val games_3 = (games_2_5.join(coaches, games_2_5.col("Season") === coaches.col("Season")
                           && games_2_5.col("WTeamID") === coaches.col("TeamID")
                           && games_2_5.col("DayNum") >= coaches.col("FirstDayNum")
                           && games_2_5.col("DayNum") <= coaches.col("LastDayNum")
                           ,"left")
               .drop(coaches.col("Season"))
               .drop(coaches.col("TeamID"))
               .drop(coaches.col("FirstDayNum"))
               .drop(coaches.col("LastDayNum"))
               .withColumnRenamed("CoachName","WCoachName")
               )

// COMMAND ----------

val games_4 = (games_3.join(coaches, games_3.col("Season") === coaches.col("Season")
                           && games_3.col("LTeamID") === coaches.col("TeamID")
                           && games_3.col("DayNum") >= coaches.col("FirstDayNum")
                           && games_3.col("DayNum") <= coaches.col("LastDayNum")
                           ,"left")
               .drop(coaches.col("Season"))
               .drop(coaches.col("TeamID"))
               .drop(coaches.col("FirstDayNum"))
               .drop(coaches.col("LastDayNum"))
               .withColumnRenamed("CoachName","LCoachName")
               )

// COMMAND ----------

val games_4_5 = games_4.join(elo, games_4.col("Season") === elo.col("Season")
                           && games_4.col("WTeamID") === elo.col("WTeamID")
                           && games_4.col("LTeamID") === elo.col("LTeamID")
                           && games_4.col("DayNum") === elo.col("DayNum")
                           ,"left")
                        .drop(elo.col("Season"))
                        .drop(elo.col("DayNum"))
                        .drop(elo.col("WTeamID"))
                        .drop(elo.col("LTeamID"))
                        .drop(elo.col("WTeamIDNewElo"))
                        .drop(elo.col("LTeamIDNewElo"))
                        .withColumnRenamed("LTeamIDElo","LTeamElo")

// COMMAND ----------

val games_5 = games_4_5.orderBy($"Season".asc, $"DayNum".asc)

// COMMAND ----------

val games_6 = (games_5.join(stats, games_5.col("Season") === stats.col("Season")
                           && games_5.col("DayNum") === stats.col("DayNum")
                           && games_5.col("WTeamID") === stats.col("TeamID")
                          ,"left")
               .drop(stats.col("Season"))
               .drop(stats.col("DayNum"))
               .drop(stats.col("TeamID"))
               .withColumnRenamed("Outcome","WOutcome")
               .withColumnRenamed("AvgScore","WAvgScore")
               .withColumnRenamed("AvgFG","WAvgFG")
               .withColumnRenamed("AvgFG3","WAvgFG3")
               .withColumnRenamed("AvgFT","WAvgFT")
               .withColumnRenamed("AvgOR","WAvgOR")
               .withColumnRenamed("AvgDR","WAvgDR")
               .withColumnRenamed("AvgAst","WAvgAst")
               .withColumnRenamed("AvgTO","WAvgTO")
               .withColumnRenamed("AvgStl","WAvgStl")
               .withColumnRenamed("AvgBlk","WAvgBlk")
               .withColumnRenamed("AvgPF","WAvgPF")
               )

// COMMAND ----------

val games_7 = (games_6.join(stats, games_6.col("Season") === stats.col("Season")
                           && games_6.col("DayNum") === stats.col("DayNum")
                           && games_6.col("LTeamID") === stats.col("TeamID")
                          ,"left")
               .drop(stats.col("Season"))
               .drop(stats.col("DayNum"))
               .drop(stats.col("TeamID"))
               .withColumnRenamed("Outcome","LOutcome")
               .withColumnRenamed("AvgScore","LAvgScore")
               .withColumnRenamed("AvgFG","LAvgFG")
               .withColumnRenamed("AvgFG3","LAvgFG3")
               .withColumnRenamed("AvgFT","LAvgFT")
               .withColumnRenamed("AvgOR","LAvgOR")
               .withColumnRenamed("AvgDR","LAvgDR")
               .withColumnRenamed("AvgAst","LAvgAst")
               .withColumnRenamed("AvgTO","LAvgTO")
               .withColumnRenamed("AvgStl","LAvgStl")
               .withColumnRenamed("AvgBlk","LAvgBlk")
               .withColumnRenamed("AvgPF","LAvgPF")
               )

// COMMAND ----------

val games_7_1 = (games_7.join(teams, games_7.col("WTeamID") === teams.col("TeamID"),"left")
                 .drop(teams.col("TeamID"))
                 .drop(teams.col("FirstD1Season"))
                 .drop(teams.col("LastD1Season"))
                 .withColumnRenamed("TeamName","WTeamName")
                 )
val games_7_2 = (games_7_1.join(teams, games_7_1.col("LTeamID") === teams.col("TeamID"),"left")
                 .drop(teams.col("TeamID"))
                 .drop(teams.col("FirstD1Season"))
                 .drop(teams.col("LastD1Season"))
                 .withColumnRenamed("TeamName","LTeamName")
                 )

// COMMAND ----------

val games_7_3 = (games_7_2.select($"Season",$"Daynum"
                             ,$"WTeamID",$"WTeamName",$"WCoachName",$"WScore",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF",$"WTeamElo"
                             ,$"LTeamID",$"LTeamName",$"LCoachName",$"LScore",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF",$"LTeamElo"
                            )
                 )

// COMMAND ----------

val games_7_5 = games_7_3.na.drop()

// COMMAND ----------

//val wteam_indexer = new StringIndexer().setInputCol("WTeamName").setOutputCol("WTeamIndex")
//val games_7_6 = wteam_indexer.fit(games_7_5).transform(games_7_5)

// COMMAND ----------

/*
val games_8 = games_7.select($"Season",$"Daynum",$"CRType",$"CityID",$"WTeamID",$"WLoc"
                             ,$"WCoachName",$"WScore",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF"
                             ,$"LTeamID",$"LCoachName",$"LScore",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF"
                            )
*/
val games_8 = games_7_5.select($"Season",$"Daynum"
                             ,$"WTeamID",$"WTeamName",$"WCoachName",$"WScore",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF",$"WTeamElo"
                             ,$"LTeamID",$"LTeamName",$"LCoachName",$"LScore",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF",$"LTeamElo"
                            )

// COMMAND ----------

val wcoach_indexer = new StringIndexer().setInputCol("WCoachName").setOutputCol("WCoachIndex")
val lcoach_indexer = new StringIndexer().setInputCol("LCoachName").setOutputCol("LCoachIndex")
//val wloc_indexer = new StringIndexer().setInputCol("WLoc").setOutputCol("WLocIndex")
//val crtype_indexer = new StringIndexer().setInputCol("CRType").setOutputCol("CRTypeIndex")

// COMMAND ----------

val games_9 = wcoach_indexer.fit(games_8).transform(games_8)
val games_10 = lcoach_indexer.fit(games_9).transform(games_9)
//val games_11 = wloc_indexer.fit(games_10).transform(games_10)
//val games_12 = crtype_indexer.fit(games_11).transform(games_11)

// COMMAND ----------

//val wloc_encoder = new OneHotEncoder().setInputCol("WLocIndex").setOutputCol("WLocVec")
//val crtype_encoder = new OneHotEncoder().setInputCol("CRTypeIndex").setOutputCol("CRTypeVec")

// COMMAND ----------

//val games_13 = wloc_encoder.transform(games_12)
//val games_14 = crtype_encoder.transform(games_13)
val games_14 = games_10

// COMMAND ----------

val games_W = games_14.withColumn("label",$"WScore")
val games_L = games_14.withColumn("label",$"LScore")

// COMMAND ----------

/*
val assembler = new VectorAssembler()
                    .setInputCols(Array("Season","Daynum","CRTypeVec","CityID","WTeamID","WCoachNameIndex"
                                        ,"WLocVec","WAvgScore","WAvgFG","WAvgFG3","WAvgFT","WAvgOR","WAvgDR"
                                        ,"WAvgAst","WAvgTO","WAvgStl","WAvgBlk","WAvgPF"
                                        ,"LTeamID","LCoachNameIndex"
                                        ,"LAvgScore","LAvgFG","LAvgFG3","LAvgFT","LAvgOR","LAvgDR"
                                        ,"LAvgAst","LAvgTO","LAvgStl","LAvgBlk","LAvgPF"))
                    .setOutputCol("features")
*/
val assembler = new VectorAssembler()
                    .setInputCols(Array("Season","Daynum"
                                        ,"WCoachIndex","WAvgScore","WAvgFG","WAvgFG3","WAvgFT"
                                        ,"WAvgOR","WAvgDR","WAvgAst","WAvgTO","WAvgStl","WAvgBlk","WAvgPF","WTeamElo"
                                        ,"LCoachIndex","LAvgScore","LAvgFG","LAvgFG3","LAvgFT"
                                        ,"LAvgOR","LAvgDR","LAvgAst","LAvgTO","LAvgStl","LAvgBlk","LAvgPF","LTeamElo"))
                    .setOutputCol("features")
val w_output = assembler.transform(games_W)
val l_output = assembler.transform(games_L)

// COMMAND ----------

val w_games_final = w_output.select($"label",$"features")
val l_games_final = l_output.select($"label",$"features")

// COMMAND ----------

//val numClasses = w_games_final.select("label").distinct.count()

// COMMAND ----------

val Array(w_split30, w_split70) = w_games_final.randomSplit(Array(0.30, 0.70), 1800009193L)
val w_testSet = w_split30.cache()
val w_trainingSet = w_split70.cache()

val Array(l_split30, l_split70) = l_games_final.randomSplit(Array(0.30, 0.70), 1800009193L)
val l_testSet = l_split30.cache()
val l_trainingSet = l_split70.cache()

// COMMAND ----------

val paramMap = List(
  "eta" -> 0.1f,
  "max_depth" -> 10,
  //"num_class" -> numClasses,
  //"objective" -> "multi:softprob"
  "objective" -> "reg:linear"
).toMap

// COMMAND ----------

val w_xgboostModelDF = XGBoost.trainWithDataFrame(w_games_final, paramMap, 50, 2, useExternalMemory=true)
val l_xgboostModelDF = XGBoost.trainWithDataFrame(l_games_final, paramMap, 50, 2, useExternalMemory=true)

// COMMAND ----------

val w_predictions = w_xgboostModelDF.transform(w_testSet)
val l_predictions = l_xgboostModelDF.transform(l_testSet)

// COMMAND ----------

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val w_rmse = evaluator.evaluate(w_predictions)
print ("Winning Root mean squared error: " + w_rmse)

val l_rmse = evaluator.evaluate(l_predictions)
print ("Losing Root mean squared error: " + l_rmse)

// COMMAND ----------

val first_round = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/Round_1.csv")

// COMMAND ----------

val one_first_round_1 = first_round.withColumn("CRType",lit("NCAA"))
                     .withColumnRenamed("TeamID_1","WTeamID") 
                     .withColumnRenamed("TeamID_2","LTeamID") 
                      .drop("Team_1")
                      .drop("Team_2")

val two_first_round_1 = first_round.withColumn("CRType",lit("NCAA"))
                     .withColumnRenamed("TeamID_2","WTeamID") 
                     .withColumnRenamed("TeamID_1","LTeamID") 
                      .drop("Team_1")
                      .drop("Team_2")

// COMMAND ----------

val one_first_round_1_1 = (one_first_round_1.join(teams,one_first_round_1.col("WTeamID") === teams.col("TeamID"),"inner")
                       .drop("TeamID")
                       .drop("FirstD1Season")
                       .drop("LastD1Season")
                       .withColumnRenamed("TeamName","WTeamName")
                       )

val one_first_round_1_2 = (one_first_round_1_1.join(teams,one_first_round_1_1.col("LTeamID") === teams.col("TeamID"),"inner")
                       .drop("TeamID")
                       .drop("FirstD1Season")
                       .drop("LastD1Season")
                       .withColumnRenamed("TeamName","LTeamName")
                       )

val two_first_round_1_1 = (two_first_round_1.join(teams,two_first_round_1.col("WTeamID") === teams.col("TeamID"),"inner")
                       .drop("TeamID")
                       .drop("FirstD1Season")
                       .drop("LastD1Season")
                       .withColumnRenamed("TeamName","WTeamName")
                       )

val two_first_round_1_2 = (two_first_round_1_1.join(teams,two_first_round_1_1.col("LTeamID") === teams.col("TeamID"),"inner")
                       .drop("TeamID")
                       .drop("FirstD1Season")
                       .drop("LastD1Season")
                       .withColumnRenamed("TeamName","LTeamName")
                       )

// COMMAND ----------

val max_coach = coaches.filter("Season=2018").groupBy("TeamID").agg(max("LastDayNum").alias("DayNum"))
val recent_coaches = (coaches.join(max_coach, coaches.col("Season") === "2018"
                                   && coaches.col("TeamID") === max_coach.col("TeamID")
                                   && coaches.col("LastDayNum") === max_coach.col("DayNum")
                                   ,"left")
                      .withColumn("Current",when($"DayNum".isNull,lit("N")).otherwise(lit("Y")))
                      .drop(max_coach.col("TeamID"))
                      .drop(max_coach.col("DayNum"))
                      ).filter("Current = 'Y'")

// COMMAND ----------

val one_first_round_2 = (one_first_round_1_2.join(recent_coaches, one_first_round_1_2.col("WTeamID") === recent_coaches.col("TeamID"),"left")
                         .drop($"TeamID")
                         .drop($"FirstDayNum")
                         .drop($"LastDayNum")
                         .drop($"Current")
                         .withColumnRenamed("CoachName","WCoachName")
                        )

val two_first_round_2 = (two_first_round_1_2.join(recent_coaches, two_first_round_1_2.col("WTeamID") === recent_coaches.col("TeamID"),"left")
                         .drop($"TeamID")
                         .drop($"FirstDayNum")
                         .drop($"LastDayNum")
                         .drop($"Current")
                         .withColumnRenamed("CoachName","WCoachName")
                        )

// COMMAND ----------

val one_first_round_3 = (one_first_round_2.join(recent_coaches, one_first_round_2.col("LTeamID") === recent_coaches.col("TeamID"),"left")
                         .drop($"TeamID")
                         .drop(recent_coaches.col("Season"))
                         .drop($"FirstDayNum")
                         .drop($"LastDayNum")
                         .drop($"Current")
                         .withColumnRenamed("CoachName","LCoachName")
                        )

val two_first_round_3 = (two_first_round_2.join(recent_coaches, two_first_round_2.col("LTeamID") === recent_coaches.col("TeamID"),"left")
                         .drop($"TeamID")
                         .drop(recent_coaches.col("Season"))
                         .drop($"FirstDayNum")
                         .drop($"LastDayNum")
                         .drop($"Current")
                         .withColumnRenamed("CoachName","LCoachName")
                        )

// COMMAND ----------

val elo_1 = elo.select($"Season",$"DayNum",$"WTeamID".alias("TeamID"),$"WTeamElo".alias("TeamElo"))
val elo_2 = elo.select($"Season",$"DayNum",$"LTeamID".alias("TeamID"),$"LTeamIDElo".alias("TeamElo"))
val elo_3 = elo_1.union(elo_2).filter("Season=2018")

// COMMAND ----------

val max_elo = elo_3.filter("Season=2018").groupBy("TeamID").agg(max("DayNum").alias("DayNum"))
val recent_elo = elo_3.join(max_elo,elo_3.col("TeamID")===max_elo.col("TeamID")
                            && elo_3.col("DayNum")===max_elo.col("DayNum"),"inner")
                    .drop(max_elo.col("TeamID"))
                    .drop(max_elo.col("DayNum"))

// COMMAND ----------

val one_first_round_4 = one_first_round_3.join(recent_elo,one_first_round_2.col("WTeamID")===recent_elo.col("TeamID"),"left")
                      .drop(recent_elo.col("DayNum"))
                      .drop(recent_elo.col("TeamID"))
                      .drop(recent_elo.col("Season"))
                      .withColumnRenamed("TeamElo","WTeamElo")

val two_first_round_4 = two_first_round_3.join(recent_elo,two_first_round_2.col("WTeamID")===recent_elo.col("TeamID"),"left")
                      .drop(recent_elo.col("DayNum"))
                      .drop(recent_elo.col("TeamID"))
                      .drop(recent_elo.col("Season"))
                      .withColumnRenamed("TeamElo","WTeamElo")

// COMMAND ----------

val one_first_round_5 = one_first_round_4.join(recent_elo,one_first_round_4.col("LTeamID")===recent_elo.col("TeamID"),"left")
                      .drop(recent_elo.col("DayNum"))
                      .drop(recent_elo.col("TeamID"))
                      .drop(recent_elo.col("Season"))
                      .withColumnRenamed("TeamElo","LTeamElo")

val two_first_round_5 = two_first_round_4.join(recent_elo,two_first_round_4.col("LTeamID")===recent_elo.col("TeamID"),"left")
                      .drop(recent_elo.col("DayNum"))
                      .drop(recent_elo.col("TeamID"))
                      .drop(recent_elo.col("Season"))
                      .withColumnRenamed("TeamElo","LTeamElo")

// COMMAND ----------

val max_stats = stats.filter("Season=2018").groupBy("TeamID").agg(max("DayNum").alias("DayNum"))
val recent_stats = stats.filter("Season=2018").join(max_stats,stats.col("TeamID")===max_stats.col("TeamID")
                            && stats.col("DayNum")===max_stats.col("DayNum"),"inner")
                    .drop(max_stats.col("TeamID"))
                    .drop(max_stats.col("DayNum"))

// COMMAND ----------

val one_first_round_6 = (one_first_round_5.join(recent_stats, one_first_round_5.col("Season") === recent_stats.col("Season")
                           //&& first_round_5.col("DayNum") === recent_stats.col("DayNum")
                           && one_first_round_5.col("WTeamID") === recent_stats.col("TeamID")
                          ,"left")
               .drop(recent_stats.col("Season"))
               .drop(recent_stats.col("DayNum"))
               .drop(recent_stats.col("TeamID"))
               .withColumnRenamed("Outcome","WOutcome")
               .withColumnRenamed("AvgScore","WAvgScore")
               .withColumnRenamed("AvgFG","WAvgFG")
               .withColumnRenamed("AvgFG3","WAvgFG3")
               .withColumnRenamed("AvgFT","WAvgFT")
               .withColumnRenamed("AvgOR","WAvgOR")
               .withColumnRenamed("AvgDR","WAvgDR")
               .withColumnRenamed("AvgAst","WAvgAst")
               .withColumnRenamed("AvgTO","WAvgTO")
               .withColumnRenamed("AvgStl","WAvgStl")
               .withColumnRenamed("AvgBlk","WAvgBlk")
               .withColumnRenamed("AvgPF","WAvgPF")
               )

val two_first_round_6 = (two_first_round_5.join(recent_stats, two_first_round_5.col("Season") === recent_stats.col("Season")
                           //&& first_round_5.col("DayNum") === recent_stats.col("DayNum")
                           && two_first_round_5.col("WTeamID") === recent_stats.col("TeamID")
                          ,"left")
               .drop(recent_stats.col("Season"))
               .drop(recent_stats.col("DayNum"))
               .drop(recent_stats.col("TeamID"))
               .withColumnRenamed("Outcome","WOutcome")
               .withColumnRenamed("AvgScore","WAvgScore")
               .withColumnRenamed("AvgFG","WAvgFG")
               .withColumnRenamed("AvgFG3","WAvgFG3")
               .withColumnRenamed("AvgFT","WAvgFT")
               .withColumnRenamed("AvgOR","WAvgOR")
               .withColumnRenamed("AvgDR","WAvgDR")
               .withColumnRenamed("AvgAst","WAvgAst")
               .withColumnRenamed("AvgTO","WAvgTO")
               .withColumnRenamed("AvgStl","WAvgStl")
               .withColumnRenamed("AvgBlk","WAvgBlk")
               .withColumnRenamed("AvgPF","WAvgPF")
               )

// COMMAND ----------

val one_first_round_7 = (one_first_round_6.join(recent_stats, one_first_round_6.col("Season") === recent_stats.col("Season")
                           && one_first_round_6.col("LTeamID") === recent_stats.col("TeamID")
                          ,"left")
               .drop(recent_stats.col("Season"))
               .drop(recent_stats.col("DayNum"))
               .drop(recent_stats.col("TeamID"))
               .withColumn("WLoc",lit("N"))
               .withColumn("WScore",lit("0"))
               .withColumn("LScore",lit("0"))
               .withColumnRenamed("Outcome","LOutcome")
               .withColumnRenamed("AvgScore","LAvgScore")
               .withColumnRenamed("AvgFG","LAvgFG")
               .withColumnRenamed("AvgFG3","LAvgFG3")
               .withColumnRenamed("AvgFT","LAvgFT")
               .withColumnRenamed("AvgOR","LAvgOR")
               .withColumnRenamed("AvgDR","LAvgDR")
               .withColumnRenamed("AvgAst","LAvgAst")
               .withColumnRenamed("AvgTO","LAvgTO")
               .withColumnRenamed("AvgStl","LAvgStl")
               .withColumnRenamed("AvgBlk","LAvgBlk")
               .withColumnRenamed("AvgPF","LAvgPF")
               )

val two_first_round_7 = (two_first_round_6.join(recent_stats, two_first_round_6.col("Season") === recent_stats.col("Season")
                           && two_first_round_6.col("LTeamID") === recent_stats.col("TeamID")
                          ,"left")
               .drop(recent_stats.col("Season"))
               .drop(recent_stats.col("DayNum"))
               .drop(recent_stats.col("TeamID"))
               .withColumn("WLoc",lit("N"))
               .withColumn("WScore",lit("0"))
               .withColumn("LScore",lit("0"))
               .withColumnRenamed("Outcome","LOutcome")
               .withColumnRenamed("AvgScore","LAvgScore")
               .withColumnRenamed("AvgFG","LAvgFG")
               .withColumnRenamed("AvgFG3","LAvgFG3")
               .withColumnRenamed("AvgFT","LAvgFT")
               .withColumnRenamed("AvgOR","LAvgOR")
               .withColumnRenamed("AvgDR","LAvgDR")
               .withColumnRenamed("AvgAst","LAvgAst")
               .withColumnRenamed("AvgTO","LAvgTO")
               .withColumnRenamed("AvgStl","LAvgStl")
               .withColumnRenamed("AvgBlk","LAvgBlk")
               .withColumnRenamed("AvgPF","LAvgPF")
               )

// COMMAND ----------

val one_first_round_7_5 = one_first_round_7.withColumn("DayNum",lit("134").cast(IntegerType))
val two_first_round_7_5 = two_first_round_7.withColumn("DayNum",lit("134").cast(IntegerType))

// COMMAND ----------

/*
val first_round_8 = first_round_7.select($"Season",$"Daynum",$"CRType",$"WTeamID",$"WLoc"
                             ,$"WCoachName",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF",$"WTeamElo"
                             ,$"LTeamID",$"LCoachName",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF",$"LTeamElo"
                            )
*/
val one_first_round_8 = one_first_round_7_5.select($"Season",$"Daynum"
                             ,$"WTeamID",$"WTeamName",$"WCoachName",$"WScore",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF",$"WTeamElo"
                             ,$"LTeamID",$"LTeamName",$"LCoachName",$"LScore",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF",$"LTeamElo"
                            )
val two_first_round_8 = two_first_round_7_5.select($"Season",$"Daynum"
                             ,$"WTeamID",$"WTeamName",$"WCoachName",$"WScore",$"WAvgScore",$"WAvgFG",$"WAvgFG3",$"WAvgFT"
                             ,$"WAvgOR",$"WAvgDR",$"WAvgAst",$"WAvgTO",$"WAvgStl",$"WAvgBlk",$"WAvgPF",$"WTeamElo"
                             ,$"LTeamID",$"LTeamName",$"LCoachName",$"LScore",$"LAvgScore",$"LAvgFG",$"LAvgFG3",$"LAvgFT"
                             ,$"LAvgOR",$"LAvgDR",$"LAvgAst",$"LAvgTO",$"LAvgStl",$"LAvgBlk",$"LAvgPF",$"LTeamElo"
                            )

// COMMAND ----------

println(one_first_round_8.na.drop().count())
println(two_first_round_8.na.drop().count())

// COMMAND ----------

val one_first_round_9 = wcoach_indexer.fit(one_first_round_8).transform(one_first_round_8)
val one_first_round_10 = lcoach_indexer.fit(one_first_round_9).transform(one_first_round_9)

val two_first_round_9 = wcoach_indexer.fit(two_first_round_8).transform(two_first_round_8)
val two_first_round_10 = lcoach_indexer.fit(two_first_round_9).transform(two_first_round_9)

// COMMAND ----------

val one_output = assembler.transform(one_first_round_10)
val two_output = assembler.transform(two_first_round_10)

// COMMAND ----------

val w_one_predictions = w_xgboostModelDF.transform(one_output)
val l_one_predictions = l_xgboostModelDF.transform(one_output)

val w_two_predictions = w_xgboostModelDF.transform(two_output)
val l_two_predictions = l_xgboostModelDF.transform(two_output)

// COMMAND ----------

val w_one = w_one_predictions.select($"WTeamID",$"WTeamName",$"LTeamID",$"LTeamName",$"Prediction".alias("WTeamScore"))
val l_one = l_one_predictions.select($"WTeamID",$"WTeamName",$"LTeamID",$"LTeamName",$"Prediction".alias("LTeamScore"))

val w_two = w_two_predictions.select($"WTeamID",$"WTeamName",$"LTeamID",$"LTeamName",$"Prediction".alias("WTeamScore"))
val l_two = l_two_predictions.select($"WTeamID",$"WTeamName",$"LTeamID",$"LTeamName",$"Prediction".alias("LTeamScore"))

// COMMAND ----------

val one = (w_one.join(l_one,w_one.col("WTeamID")===l_one.col("WTeamID")
                    && w_one.col("LTeamID")===l_one.col("LTeamID"),"inner")
           .drop(l_one.col("WTeamID"))
           .drop(l_one.col("WTeamName"))
           .drop(l_one.col("LTeamID"))
           .drop(l_one.col("LTeamName"))
           )

val two = (w_two.join(l_two,w_two.col("WTeamID")===l_two.col("WTeamID")
                    && w_two.col("LTeamID")===l_two.col("LTeamID"),"inner")
           .drop(l_two.col("WTeamID"))
           .drop(l_two.col("WTeamName"))
           .drop(l_two.col("LTeamID"))
           .drop(l_two.col("LTeamName"))
           )

// COMMAND ----------

one
   .coalesce(1)
   .write.format("com.databricks.spark.csv")
   .option("header", "true")
   .save("dbfs:/FileStore/tables/one/one.csv")

two
   .coalesce(1)
   .write.format("com.databricks.spark.csv")
   .option("header", "true")
   .save("dbfs:/FileStore/tables/two/two.csv")

// COMMAND ----------

one.show(100)

// COMMAND ----------

two.show(100)
