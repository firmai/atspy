---
title: 'AtsPy: Automated Time Series Forecasting in Python'
tags:
  - Automated
  - Time Series
  - Machine Learning
  - Python
authors:
  - name: Derek Snow
    orcid: 0000-0001-6681-6828
    affiliation: "Alan Turing Institute"
affiliations:
 - name: Research Associate, The Alan Turing Institute
   index: 1
date: 22 April 2020
bibliography: paper.bib

---

# Summary

AtsPy, an open source automated time series framework is developed as a working 
prototype to showcase the ability of state of the art univariate time series methods.


A Python-centric view on the recent growth of time series tools shows the development
of the Prophet by Facebook[@Taylor:2017], the GluonTS[@Alexandrov:2019] toolkit by Amazon, and the ForecastTCN algorithm
by Microsoft. Among others, these tools have put forecasting methods in the hands of the
everyday user. We have also seen contributions from academia and independent developers 
with algorithms and packages like N-Beats, Auto-Arima, and TBATS[@Hyndman:2011],. The majority of these 
have been implemented in AtsPy which is hosted on GitHub.The recent surge in automated 
time series methods is the direct result of new algorithms (TBATS), procedures (Prophet),
and frameworks (GluonTS). In the following section we will seek to understand how these 
methods have led to the automation of time series forecasting and also discuss how existing 
method can be used to automate predictions. 

Can be seen as a univariate instantiation of GluonTS with an emphasis on model diversity. 
AtsPy is built on top of Auto-Arima, TBATS, Prophet, and GluonTS. It is an extremely fast
method to test which model best fits your data. AtsPy's final innovation is
an ensemble time series protocol developed with the LightGBM flavour Gradient Boosting Machine
and extracted time series features. 


