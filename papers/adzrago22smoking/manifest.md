Variables:

Text comes from the paper

Use ICPSR to look for relevant variables to get variable code and directory/file.

We copied that result here.

Predictors:
Time in Years 
We constructed this variable to indicate the time in years that the data were collected from the study participants at each wave or observation (Time = 0 [1995], Time = 1 [1996], Time = 6 [2001], and Time = 13 [2008]). The following predictors were measured at all waves.

computed implicitly

From Paper:

Marijuana Use 
Frequency Marijuana use was measured by asking the participants to indicate, “During the past 30 days, how many times did you use marijuana?”.


From ICPSR:

H3TO110

S28Q110 PAST 30 DAYS USED MARIJUANA -W3

During the past 30 days, how many times have you used marijuana?

Taken from: National Longitudinal Study of Adolescent to Adult Health (Add Health), 1994-2018 [Public Use].

numeric

H3TO110 - DS8
H1TO32 - DS1
H2TO46 - DS5
H4TO71 - DS22



Alcohol Use Frequency 
This variable was also measured as, “During the past 12 months, on how many days did you drink alcohol?”.


H2TO19
S27Q19 PAST 12 MOS-DAYS DRINK ALCOHOL-W2

During the past 12 months, on how many days did you drink alcohol?

Taken from: National Longitudinal Study of Adolescent to Adult Health (Add Health), 1994-2018 [Public Use].
numeric
DS5

H2TO19 - DS5
H1TO15 - DS1
H4TO35 - DS22
H3TO38 - DS8



Illicit Drug Use The participants were asked “How old were you when you first tried any other type of illicit drugs, such as LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills, without a doctor’s prescription? If you never tried any other type of illicit drug, enter “0.” This variable was dichotomized (tried any other illicit drug = 1 and never tried any illicit drug = 0).

H1TO40
S28Q40 AGE FIRST OTHER ILLEGAL DRUGS-W1

How old were you when you first tried any other type of illegal drug, such as LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills, without a doctor's prescription? If you never tried any other type of illegal drug, enter "0."

Taken from: National Longitudinal Study of Adolescent to Adult Health (Add Health), 1994-2018 [Public Use].
numeric

DS1

H4TO65E - DS22
H1TO40 - DS1
H3TO117 - DS8
H2TO58 - DS5


Aggressive Behavior This variable was determined by ask- ing, “In the past 12 months, how often did you deliberately damage property that didn’t belong to you?”.

H4DS1 - DS22
S21Q1 12 MO,OFT DAMAGE PROP/NOT YOUR-W4

In the past 12 months, how often did you deliberately damage property that didn't belong to you?

Taken from: National Longitudinal Study of Adolescent to Adult Health (Add Health), 1994-2018 [Public Use].
numeric
DS22


H3DS1 - DS8
S26Q1 12 MO,OFT DAMAGE PROP/NOT YOUR-W3

In the past 12 months, how often did you deliberately damage property that didn't belong to you?

Taken from: National Longitudinal Study of Adolescent to Adult Health (Add Health), 1994-2018 [Public Use].
numeric
DS8

H4DS1 - DS22
H3DS1 - DS8
H2DS2 - DS5
H1DS2 - DS1

Covariates:

School attendance (If SCHOOL YEAR: Are you presently in school? If SUMMER: Were you in school during this past school year? Yes = 1 and no = 0), 

H3ED23 - DS8
H2GI6 - DS5
H1GI18 - DS1
H4ED6 - DS22

age
S1 - DS1

gender (male or female)
BIO_SEX3 - DS8

race (Hispanic or Latino = 1, non-Hispanic Black or African American = 2, non-Hispanic Asian or Pacific Islander = 3, non-Hispanic American Indian or Native American = 4, non-Hispanic Other = 5, and non- Hispanic White)

S6C - DS1 (asian)
S6B - DS1 (black)
S6A - DS1 (white)
S6D - DS1 (native american)
S6E - DS1 (other)

TARGET VARIABLE

During the past 30 days, on the days you smoked, how many cigarettes did you smoke each day?

H3TO10 - DS8
H1TO7 - DS1
H2TO7 - DS5
H4TO6 - DS22