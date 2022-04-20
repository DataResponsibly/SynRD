#!/bin/bash
awk -vFPAT='([^,]*)|("[^"]+")' -vOFS=, 'BEGIN{headerNF = 0} {if (NR==1) {print "Header has " NF " fields."; headerNF = NF}; if (NF != headerNF) {print "Row " NR " has " NF " fields, not " headerNF}}' $1
