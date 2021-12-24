#!/bin/bash

str="1"
while (( ${#str2} < 640 ))
do
    str2="${str2}${str}"
    curl --header "Content-Type: application/json" --request POST --data '{"context": '\""${str2}"\"', "n": 8}' http://localhost:5000/complete
done