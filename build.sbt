import sbt.Resolver

name := "Alchemist"

version := "1.0"

resolvers ++= Seq(
	"Unidata maven repository" at "http://artifacts.unidata.ucar.edu/content/repositories/unidata-releases",
	"Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  	"Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

lazy val commonSettings = Seq(
  organization := "amplab.alchemist",
  version := "0.0.2",
  scalaVersion := "2.11.8",
  libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % "provided",
  libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0" % "provided",
  libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % "provided", 
  test in assembly := {}
)

lazy val root = (project in file(".")).
  aggregate(core, tests)

lazy val core = (project in file("core")).
  settings(commonSettings: _*).
  settings(
    name := "alchemist"
  )

lazy val tests = (project in file("test")).
  settings(commonSettings: _*).
  settings(
    name := "alchemist-tests",
    mainClass in assembly := Some("amplab.alchemist.BasicSuite")
  ).
  dependsOn(core)
