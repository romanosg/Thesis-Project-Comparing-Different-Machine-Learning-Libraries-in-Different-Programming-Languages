import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.5.10"
    application
}

group = "me.georg"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))

    // library installation
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.4.0-alpha-2") // KotlinDL

    implementation("com.github.haifengl:smile-kotlin:2.6.0") // Smile
    implementation("org.slf4j:slf4j-api:1.7.5") // Smile dataframe
    implementation("org.slf4j:slf4j-log4j12:1.7.5")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClass.set("MainKt")
}