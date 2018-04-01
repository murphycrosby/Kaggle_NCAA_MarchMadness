// Databricks notebook source
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import sqlContext.implicits._

// COMMAND ----------

val K = 22.0
val HOME_ADVANTAGE = 100.0

// COMMAND ----------

def elo_pred(elo1:Double, elo2:Double): Double = {
  return (1.0 / (scala.math.pow(10.0,(-(elo1 - elo2) / 400.0)) + 1.0))
}

def expected_margin(elo_diff:Double): Double = {
  return((7.5 + 0.006 * elo_diff))
}

def elo_update(w_elo:Double, l_elo:Double, margin:Int): (Double, Double) = {
  val elo_diff = w_elo - l_elo
  val pred = elo_pred(w_elo, l_elo)
  val mult = scala.math.pow((margin + 3.0),0.8) / expected_margin(elo_diff)
  val update = K * mult * (1 - pred)
  
  return(pred,update)
}

// COMMAND ----------

def eloDataframeToMap(df:DataFrame): collection.mutable.Map[Int,Double] = {
  val elo = collection.mutable.Map[Int,Double]()
  for(row <- df.collect()) {
    elo += row.getInt(0) -> row.getDouble(1)
  }
  return elo
}

// COMMAND ----------

def set_elo(teamDF:DataFrame, eloMap:collection.mutable.Map[Int,Double], init:Boolean): collection.mutable.Map[Int,Double] = {
  val team_elo = collection.mutable.Map[Int,Double]()
  
  if(init == true) {
    for(row <- teamDF.collect()) {
      team_elo += (row.getInt(0) -> 1500.0)
    }
  } else {
    for(row <- teamDF.collect()) {
      val e = (eloMap(row.getInt(0)) * .75) + (.25 * 1505)
      team_elo += row.getInt(0) -> e
    }
  }
  return team_elo
}

// COMMAND ----------

def calc_elo(gameRow:Row, elo_map:collection.mutable.Map[Int,Double]) : (Double,Double,Int,Double,Double,Int,Double,Double) = {
  val new_elo_map = elo_map
  val row_season = gameRow.getInt(0)
  val numDay = gameRow.getInt(1)
  val wID = gameRow.getInt(2)
  val lID = gameRow.getInt(4)
  val margin = gameRow.getInt(3) - gameRow.getInt(5)
  val wLoc = gameRow.getString(6)  

  var wAdd = 0.0
  var lAdd = 0.0
    
  if(wLoc == "H") {
    wAdd += HOME_ADVANTAGE
  } else if (wLoc == "A") {
    lAdd += HOME_ADVANTAGE
  }
  
  val welo = new_elo_map(wID)
  val lelo = new_elo_map(lID)
  
  val(pred,update) = elo_update(welo+wAdd, lelo+lAdd, margin)
  
  return (pred,update,wID,welo,(welo+update),lID,lelo,(lelo-update))
}

// COMMAND ----------

def calc_elo_year(season:Int,d_frame:DataFrame,elo_map:collection.mutable.Map[Int,Double]) : (collection.mutable.Map[Int,Double]) = {
  dbutils.fs.rm("dbfs:/FileStore/tables/"+season+"_EloRatings/",true)
  dbutils.fs.rm("dbfs:/FileStore/tables/Final_"+season+"_EloRatings/",true)
  
  val map = elo_map
  val days = d_frame.agg(max("DayNum")).first.getInt(0)
  
  for (i <- 1 to days) {
    val df = d_frame.filter($"DayNum" === i)
    if(df.count > 0) {
      val dayDF = (df.map(r => {calc_elo(r,map)})
                   .toDF("Prediction","Update","WTeamID","WTeamElo","WTeamIDNewElo","LTeamID","LTeamIDElo","LTeamIDNewElo")
                   .withColumn("Season",lit(season))
                   .withColumn("DayNum",lit(i))
                   .select($"Season",$"DayNum",$"Prediction",$"Update"
                           ,$"WTeamID",$"WTeamElo",$"WTeamIDNewElo"
                           ,$"LTeamID",$"LTeamIDElo",$"LTeamIDNewElo")
                  )
      dayDF.write.mode(SaveMode.Append).format("orc").save("dbfs:/FileStore/tables/"+season+"_EloRatings/")
      
      dayDF.collect().map(r => {
        map += (r.getInt(4) -> r.getDouble(6))
        map += (r.getInt(7) -> r.getDouble(9))
      })
    }
  }
  
  val team_elo_df = map.toSeq.toDF("TeamID", "FinalElo")
  team_elo_df.write.mode(SaveMode.Overwrite).format("orc").save("dbfs:/FileStore/tables/Final_"+season+"_EloRatings/")
  
  return map
}

// COMMAND ----------

val teams = spark.read.format("csv")
                            .option("header", "true")
                            .option("inferSchema", "true")
                            .load("dbfs:/FileStore/tables/Teams.csv")
val ncaaresults = spark.read.format("csv")
                            .option("header", "true")
                            .option("inferSchema", "true")
                            .load("dbfs:/FileStore/tables/NCAATourneyDetailedResults.csv")
val regresults = spark.read.format("csv")
                            .option("header", "true")
                            .option("inferSchema","true")
                            .load("dbfs:/FileStore/tables/RegularSeasonDetailedResults.csv")
val regresults_2018 = spark.read.format("csv")
                            .option("header", "true")
                            .option("inferSchema","true")
                            .load("dbfs:/FileStore/tables/RegularSeasonDetailedResults_2018.csv")

// COMMAND ----------

val games_1 = regresults.union(ncaaresults).union(regresults_2018)

// COMMAND ----------

val games_5 = games_1.orderBy($"Season".asc, $"DayNum".asc)

// COMMAND ----------

//2003
val games_2003 = games_5.filter($"Season" === 2003)
val team_elo_2003_before = set_elo(teams, null, true)
//val team_elo_2003_after = calc_elo_year(2003,games_2003,team_elo_2003_before)
val ratings_2003 = spark.read.orc("dbfs:/FileStore/tables/2003_EloRatings/")
val final_ratings_2003 = spark.read.orc("dbfs:/FileStore/tables/Final_2003_EloRatings/")
val team_elo_2003_after = eloDataframeToMap(final_ratings_2003)
val pred_2003 = ratings_2003.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2004
val games_2004 = games_5.filter($"Season" === 2004)
val team_elo_2004_before = set_elo(teams, team_elo_2003_after, false)
//val team_elo_2004_after = calc_elo_year(2004,games_2004,team_elo_2004_before)
val ratings_2004 = spark.read.orc("dbfs:/FileStore/tables/2004_EloRatings/")
val final_ratings_2004 = spark.read.orc("dbfs:/FileStore/tables/Final_2004_EloRatings/")
val team_elo_2004_after = eloDataframeToMap(final_ratings_2004)
val pred_2004 = ratings_2004.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2005
val games_2005 = games_5.filter($"Season" === 2005)
val team_elo_2005_before = set_elo(teams, team_elo_2004_after, false)
//val team_elo_2005_after = calc_elo_year(2005,games_2005,team_elo_2005_before)
val ratings_2005 = spark.read.orc("dbfs:/FileStore/tables/2005_EloRatings/")
val final_ratings_2005 = spark.read.orc("dbfs:/FileStore/tables/Final_2005_EloRatings/")
val team_elo_2005_after = eloDataframeToMap(final_ratings_2005)
val pred_2005 = ratings_2005.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2006
val games_2006 = games_5.filter($"Season" === 2006)
val team_elo_2006_before = set_elo(teams, team_elo_2005_after, false)
//val team_elo_2006_after = calc_elo_year(2006,games_2006,team_elo_2006_before)
val ratings_2006 = spark.read.orc("dbfs:/FileStore/tables/2006_EloRatings/")
val final_ratings_2006 = spark.read.orc("dbfs:/FileStore/tables/Final_2006_EloRatings/")
val team_elo_2006_after = eloDataframeToMap(final_ratings_2006)
val pred_2006 = ratings_2006.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2007
val games_2007 = games_5.filter($"Season" === 2007)
val team_elo_2007_before = set_elo(teams, team_elo_2006_after, false)
//val team_elo_2007_after = calc_elo_year(2007,games_2007,team_elo_2007_before)
val ratings_2007 = spark.read.orc("dbfs:/FileStore/tables/2007_EloRatings/")
val final_ratings_2007 = spark.read.orc("dbfs:/FileStore/tables/Final_2007_EloRatings/")
val team_elo_2007_after = eloDataframeToMap(final_ratings_2007)
val pred_2007 = ratings_2007.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2008
val games_2008 = games_5.filter($"Season" === 2008)
val team_elo_2008_before = set_elo(teams, team_elo_2007_after, false)
val team_elo_2008_after = calc_elo_year(2008,games_2008,team_elo_2008_before)
val ratings_2008 = spark.read.orc("dbfs:/FileStore/tables/2008_EloRatings/")
//val final_ratings_2008 = spark.read.orc("dbfs:/FileStore/tables/Final_2008_EloRatings/")
//val team_elo_2008_after = eloDataframeToMap(final_ratings_2008)
val pred_2008 = ratings_2008.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

pred_2007

// COMMAND ----------

//2009
val games_2009 = games_5.filter($"Season" === 2009)
val team_elo_2009_before = set_elo(teams, team_elo_2008_after, false)
val team_elo_2009_after = calc_elo_year(2009,games_2009,team_elo_2009_before)
val ratings_2009 = spark.read.orc("dbfs:/FileStore/tables/2009_EloRatings/")
//val final_ratings_2009 = spark.read.orc("dbfs:/FileStore/tables/Final_2009_EloRatings/")
//val team_elo_2009_after = eloDataframeToMap(final_ratings_2009)
val pred_2009 = ratings_2009.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2010
val games_2010 = games_5.filter($"Season" === 2010)
val team_elo_2010_before = set_elo(teams, team_elo_2009_after, false)
val team_elo_2010_after = calc_elo_year(2010,games_2010,team_elo_2010_before)
val ratings_2010 = spark.read.orc("dbfs:/FileStore/tables/2010_EloRatings/")
//val final_ratings_2010 = spark.read.orc("dbfs:/FileStore/tables/Final_2010_EloRatings/")
//val team_elo_2010_after = eloDataframeToMap(final_ratings_2010)
val pred_2010 = ratings_2010.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2011
val games_2011 = games_5.filter($"Season" === 2011)
val team_elo_2011_before = set_elo(teams, team_elo_2010_after, false)
val team_elo_2011_after = calc_elo_year(2011,games_2011,team_elo_2011_before)
val ratings_2011 = spark.read.orc("dbfs:/FileStore/tables/2011_EloRatings/")
//val final_ratings_2011 = spark.read.orc("dbfs:/FileStore/tables/Final_2011_EloRatings/")
//val team_elo_2011_after = eloDataframeToMap(final_ratings_2011)
val pred_2011 = ratings_2011.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2012
val games_2012 = games_5.filter($"Season" === 2012)
val team_elo_2012_before = set_elo(teams, team_elo_2011_after, false)
//val team_elo_2012_after = calc_elo_year(2012,games_2012,team_elo_2012_before)
val ratings_2012 = spark.read.orc("dbfs:/FileStore/tables/2012_EloRatings/")
//val final_ratings_2012 = spark.read.orc("dbfs:/FileStore/tables/Final_2012_EloRatings/")
//val team_elo_2012_after = eloDataframeToMap(final_ratings_2012)
val pred_2012 = ratings_2012.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2013
val games_2013 = games_5.filter($"Season" === 2013)
val team_elo_2013_before = set_elo(teams, team_elo_2012_after, false)
val team_elo_2013_after = calc_elo_year(2013,games_2013,team_elo_2013_before)
val ratings_2013 = spark.read.orc("dbfs:/FileStore/tables/2013_EloRatings/")
//val final_ratings_2013 = spark.read.orc("dbfs:/FileStore/tables/Final_2013_EloRatings/")
//val team_elo_2013_after = eloDataframeToMap(final_ratings_2013)
val pred_2013 = ratings_2013.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2014
val games_2014 = games_5.filter($"Season" === 2014)
val team_elo_2014_before = set_elo(teams, team_elo_2013_after, false)
val team_elo_2014_after = calc_elo_year(2014,games_2014,team_elo_2014_before)
val ratings_2014 = spark.read.orc("dbfs:/FileStore/tables/2014_EloRatings/")
//val final_ratings_2014 = spark.read.orc("dbfs:/FileStore/tables/Final_2014_EloRatings/")
//val team_elo_2014_after = eloDataframeToMap(final_ratings_2014)
val pred_2014 = ratings_2014.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2015
val games_2015 = games_5.filter($"Season" === 2015)
val team_elo_2015_before = set_elo(teams, team_elo_2014_after, false)
val team_elo_2015_after = calc_elo_year(2015,games_2015,team_elo_2015_before)
val ratings_2015 = spark.read.orc("dbfs:/FileStore/tables/2015_EloRatings/")
//val final_ratings_2015 = spark.read.orc("dbfs:/FileStore/tables/Final_2015_EloRatings/")
//val team_elo_2015_after = eloDataframeToMap(final_ratings_2015)
val pred_2015 = ratings_2015.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2016
val games_2016 = games_5.filter($"Season" === 2016)
val team_elo_2016_before = set_elo(teams, team_elo_2015_after, false)
val team_elo_2016_after = calc_elo_year(2016,games_2016,team_elo_2016_before)
val ratings_2016 = spark.read.orc("dbfs:/FileStore/tables/2016_EloRatings/")
//val final_ratings_2016 = spark.read.orc("dbfs:/FileStore/tables/Final_2016_EloRatings/")
//val team_elo_2016_after = eloDataframeToMap(final_ratings_2016)
val pred_2016 = ratings_2016.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2017
val games_2017 = games_5.filter($"Season" === 2017)
val team_elo_2017_before = set_elo(teams, team_elo_2016_after, false)
val team_elo_2017_after = calc_elo_year(2017,games_2017,team_elo_2017_before)
val ratings_2017 = spark.read.orc("dbfs:/FileStore/tables/2017_EloRatings/")
//val final_ratings_2017 = spark.read.orc("dbfs:/FileStore/tables/Final_2017_EloRatings/")
//val team_elo_2017_after = eloDataframeToMap(final_ratings_2017)
val pred_2017 = ratings_2017.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

//2018
val games_2018 = games_5.filter($"Season" === 2018)
val team_elo_2018_before = set_elo(teams, team_elo_2017_after, false)
val team_elo_2018_after = calc_elo_year(2018,games_2018,team_elo_2018_before)
val ratings_2018 = spark.read.orc("dbfs:/FileStore/tables/2018_EloRatings/")
//val final_ratings_2018 = spark.read.orc("dbfs:/FileStore/tables/Final_2018_EloRatings/")
//val team_elo_2018_after = eloDataframeToMap(final_ratings_2018)
val pred_2018 = ratings_2018.agg(mean(-log("Prediction"))).first.getDouble(0)

// COMMAND ----------

println(pred_2016)
println(pred_2017)
println(pred_2018)

// COMMAND ----------

//ratings_2003.count()
ratings_2018.filter("WTeamID=1328 OR LTeamID=1328").orderBy("DayNum").show(1000)

// COMMAND ----------

teams.filter("TeamID=1303").show()

// COMMAND ----------

/*
val schema = StructType(
  StructField("Season", IntegerType, true) 
  :: StructField("DayNum", IntegerType, true)
  :: StructField("Prediction", DoubleType, true)
  :: StructField("Update", DoubleType, true)
  :: StructField("WTeamID", IntegerType, true) 
  :: StructField("WTeamElo", DoubleType, true)
  :: StructField("WTeamIDNewElo", DoubleType, true)
  :: StructField("LTeamID", IntegerType, true)
  :: StructField("LTeamIDElo", DoubleType, true)
  :: StructField("LTeamIDNewElo", DoubleType, true)
  :: Nil
)
var ratings = spark.createDataFrame(sc.emptyRDD[Row], schema)

val days = games_2003.agg(max("DayNum")).first.getInt(0)

for (i <- 1 to days) {
  val df = games_2003.filter($"DayNum" === i)
  if(df.count > 0) {
    val dayDF = (df.map(r => {calc_elo(r,team_elo_2003)})
                  .toDF("Prediction","Update","WTeamID","WTeamElo","WTeamIDNewElo","LTeamID","LTeamIDElo","LTeamIDNewElo")
                  .withColumn("Season",lit(2003))
                  .withColumn("DayNum",lit(i))
                  .select($"Season",$"DayNum",$"Prediction",$"Update"
                          ,$"WTeamID",$"WTeamElo",$"WTeamIDNewElo"
                          ,$"LTeamID",$"LTeamIDElo",$"LTeamIDNewElo")
                 )
    
    //dayDF.show()
    dayDF.write.mode(SaveMode.Append).format("orc").save("dbfs:/FileStore/tables/2003_EloRatings/")
    
    dayDF.collect().map(r => {
      team_elo_2003 += (r.getInt(4) -> r.getDouble(6))
      team_elo_2003 += (r.getInt(7) -> r.getDouble(9))
    })
  }
}
*/

// COMMAND ----------


