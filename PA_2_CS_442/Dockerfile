FROM ubuntu:18.04
MAINTAINER Ryan Lunt
LABEL description="Cloud computing - Machine Learning project with Spark & Docker."

#prevents inputs.
ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /home

#some dependenceies / requirements
RUN apt-get update && apt-get upgrade
RUN apt-get install -y python3-pip
RUN apt-get install -y openjdk-8-jdk
RUN pip3 install --upgrade pip
#RUN apt-get install -y wget
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip3 install pandas

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

#set up folders / files in correct place. (WIP)
#RUN wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz
#RUN mv spark-3.1.2-bin-hadoop2.7.tgz /opt/spark
#ADD /opt/spark-3.1.2-bin-hadoop2.7.tgz /
#ADD ./spark-3.1.2-bin-hadoop2.7.tgz /opt/spark
#COPY requirements.txt /tmp
#WORKDIR /tmp
#environment variables (WIP)
#ENV "export SPARK_HOME=/opt/spark" >> ~/.profile
#ENV "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.profile
#ENV "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.profile

#pyspark
RUN pip3 install pyspark

#requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY winequality-white.csv .
COPY ML_Project2.py .

#launch program automatically
CMD ["python3", "ML_Project2.py"]
