import numpy as np
import pandas as pd
import datetime
import streamlit as st
import pandas as pd

BATTERY_POWER = 3
BATTERY_CAPACITY = 6
BATTERY_EFFICIENCY = 0.92

#in hours
CHARGING_RATE = 1 * BATTERY_EFFICIENCY
DISCHARGING_RATE = 1 * BATTERY_EFFICIENCY

# Load data
print(datetime.datetime.now())

# Data columns:
# customerID, [0]
# NumberOfPanels, [1]
# Date_UTC, [2]
# Date_NZDate, [3]
# date_settlementPeriod, [4]
# load_power_kW, [5]
# pv_totalPower_kW, [6]
# price_gridExport_NZDperkWh [7]
# price_gridImport_NZDperkWh [8]
# grid_renewableFraction_pct [9]


battery_mode = 0

data_cache = {}

def getCustomerIDs():
    customerIDs = list(range(1,101))
    return customerIDs

def getCustomerData(customerID):
    if (customerID in data_cache):
        return data_cache[customerID]
    else:
        print("Loading data for customer " + str(customerID))
        user_data = np.genfromtxt(
            f"data/customer_data_{customerID}.csv",
            delimiter=",",
            skip_header=1,
            dtype="int,int,datetime64[s],datetime64[s],float,float,float,float,float,float"
        )

        # Sort data by customer id, then date
        user_data = np.sort(user_data)

        # Convert data to 2d numpy array
        user_data = np.array([list(r) for r in user_data])

        # Get rid of nan values
        numerical_data = user_data[:,[0,1,4,5,6,7]].astype(float)
        mask = np.any(np.isnan(numerical_data), axis=1)
        user_data = user_data[~mask]

        # Cache data
        data_cache[customerID] = user_data

        return user_data

def trimDataToDay(data, date):
    date_times = [np.datetime64(t,"D") for t in data[:,2]]
    data = data[date_times == np.datetime64(date,"D")]
    return data

def getDatesFromData(data):
    date_times = [np.datetime64(t,"D") for t in data[:,2]]
    dates = np.unique(date_times)
    return dates

def scalingFactorForNumberOfPanels(NumberOfPanels,CurrentNumberOfPanels):
    return NumberOfPanels/CurrentNumberOfPanels

def toCostBeforeSolar(LoadPower,PriceForGridImport):
    return PriceForGridImport*LoadPower/4

def renewableToLoadBeforeSolar(LoadPower,RenewableFraction):
    return RenewableFraction*LoadPower/4

def suppliedPV(PVPower,LoadPower):
    return np.minimum(PVPower,LoadPower)

def batteryCharge(SolarCharge, GridCharge):
    return SolarCharge+GridCharge

def batteryDischarge(SolarDischarge, GridDischarge):
    return SolarDischarge+GridDischarge

def chargeFromSolar(BatteryMode, PV_Power, LoadPower, BatteryChargeLevel):
    if (BatteryMode == 1):
        return min(max(PV_Power-LoadPower,0),(BATTERY_CAPACITY-BatteryChargeLevel)/np.sqrt(BATTERY_EFFICIENCY))
    else:
        return 0
    
def chargeFromGrid(BatteryMode, PowerInChargeMode, BatteryChargeLevel):
    if (BatteryMode == 2):
        return min(PowerInChargeMode, (BATTERY_CAPACITY-BatteryChargeLevel)/np.sqrt(BATTERY_EFFICIENCY))
    else:
        return 0

def dischargeToLoad(BatteryMode, PV_Power, LoadPower, BatteryChargeLevel):
    if (BatteryMode == 1):
        return min(max(LoadPower-PV_Power,0), BatteryChargeLevel*np.sqrt(BATTERY_EFFICIENCY))
    else:
        return 0

def dischargeToGrid(BatteryMode, PowerInDischargeMode, BatteryChargeLevel):
    if (BatteryMode == 3):
        return min(PowerInDischargeMode,BatteryChargeLevel*np.sqrt(BATTERY_EFFICIENCY))
    else:
        return 0
    
def batteryChargeLevel(previousBatteryChargeLevel, batteryCharge, batteryDischarge):
    return previousBatteryChargeLevel + batteryCharge*np.sqrt(BATTERY_EFFICIENCY)/4 - (batteryDischarge/np.sqrt(BATTERY_EFFICIENCY))/4

def batteryChargePercentage(batteryChargeLevel):
    return batteryChargeLevel/BATTERY_CAPACITY

def newGridConsumption(LoadPower, PV_PowerSuppliedToLoad, PV_PowerAfterScaling, BatteryCharge, BatteryDischarge):
    return LoadPower-PV_PowerSuppliedToLoad-PV_PowerAfterScaling+BatteryCharge-BatteryDischarge

def energyCostPostSolar(EnergyCostBeforeSolar, newGridUsage):
    return EnergyCostBeforeSolar*max(newGridUsage,0)/2

def gridRenewablePercentage(dateNtime):
    if (dateNtime.time() < datetime.time(6)):
        return 0.9
    else:
        return 0.7

def numberOfHoursToCharge(NumberOfPanels,NumberOfBatteries,PVPower):
    return NumberOfBatteries/(NumberOfPanels*PVPower)


def findFullDay(CustomerID):
    customer_data = getCustomerData(CustomerID)
    dates = getDatesFromData(customer_data)
    for d in dates:
        if (len(trimDataToDay(customer_data, d)) == 96):
            return d

    print("create average day")
    #otherwise return 

def dateHalfToSettlementPeriod(day):
    return pd.Timestamp(day[2]).hour*4 + pd.Timestamp(day[2]).minute/15

def getAverageDayArray(customerID,indexToAverage):
    customer_data = getCustomerData(customerID)
    dates = getDatesFromData(customer_data)

    days = np.array([])
    for d in dates:
        print(d)
        day_customer_data = trimDataToDay(customer_data, d)
        day = np.full((96),float('NaN'))

        for time in day_customer_data:
            i = int(dateHalfToSettlementPeriod(time))
            day[i] = time[indexToAverage]
        
        days = np.append(days,day)

    return np.nanmean(days.reshape(-1,96),axis=0)

#new
# time 0
# total power 1
# scaling scalingFactor 2
# Load power 3
# Grid Import price 4
# grid renewable fraction 5
#Electricity cost before solar 6
# Renewable to load 7
# battery mode 8
# battery charge 9
# charge from solar 10
# charge from grid 11
# battery discharge  12
# total cost after solar 13

def determineBatteryMode(array):
    return 1

def generateModelTable(customerID,numberOfBatteries,NumberOfPanels):
    customer_data = getCustomerData(customerID)
    array = []
    current_battery_charge_level = 0.5 * BATTERY_CAPACITY * numberOfBatteries
    for row in customer_data:
        date = row[2]
        pv_total_power = row[6]
        number_of_panels = row[1]
        scaling_factor = scalingFactorForNumberOfPanels(NumberOfPanels,number_of_panels)
        scaled_pv = pv_total_power*scaling_factor
        load_power = row[5]
        grid_import_price = row[8]

        try:
            grid_renewable_fraction = row[9]
        except:
            grid_renewable_fraction = 0.6
        
        cost_before_solar = toCostBeforeSolar(load_power,grid_import_price)
        renewable_to_load = renewableToLoadBeforeSolar(load_power,grid_renewable_fraction)
        supplied_pv = suppliedPV(scaled_pv,load_power)

        battery_mode = determineBatteryMode(row)
        battery_charge_from_grid = chargeFromGrid(battery_mode,1,BATTERY_CAPACITY/2)
        battery_charge_from_solar = chargeFromSolar(battery_mode,supplied_pv,load_power,BATTERY_CAPACITY/2)
        battery_charge = battery_charge_from_grid + battery_charge_from_solar
        battery_discharge_to_load = dischargeToLoad(battery_mode,supplied_pv,load_power,BATTERY_CAPACITY/2)
        battery_discharge_to_grid = dischargeToGrid(battery_mode,1,BATTERY_CAPACITY/2)
        battery_discharge = battery_discharge_to_load + battery_discharge_to_grid
        battery_charge_level = batteryChargeLevel(current_battery_charge_level,battery_charge,battery_discharge)
        battery_charge_percentage = batteryChargePercentage(battery_charge_level)
        
        total_cost_after_solar = energyCostPostSolar(cost_before_solar,newGridConsumption(load_power,supplied_pv,scaled_pv,battery_charge,battery_discharge))
        
        array.append([
            date,
            pv_total_power,
            number_of_panels,
            scaling_factor,
            scaled_pv,
            load_power,
            grid_import_price,
            grid_renewable_fraction,
            cost_before_solar,
            renewable_to_load,
            supplied_pv,
            battery_mode,
            battery_charge_from_solar,
            battery_charge_from_grid,
            battery_charge,
            battery_discharge_to_load,
            battery_discharge_to_grid,
            battery_discharge,
            battery_charge_level,
            battery_charge_percentage,
            total_cost_after_solar
        ])
        
    # array = np.array(array, dtype=[
    #     ("date", "datetime64[s]"),
    #     ("pv_total_power", "float"),
    #     ("number_of_panels", "int"),
    #     ("scaling_factor", "float"),
    #     ("scaled_pv", "float"),
    #     ("load_power", "float"),
    #     ("grid_import_price", "float"),
    #     ("grid_renewable_fraction", "float"),
    #     ("cost_before_solar", "float"),
    #     ("renewable_to_load", "float"),
    #     ("battery_mode", "int"),
    #     ("battery_charge_from_solar", "float"),
    #     ("battery_charge_from_grid", "float"),
    #     ("battery_charge", "float"),
    #     ("battery_discharge_to_load", "float"),
    #     ("battery_discharge_to_grid", "float"),
    #     ("battery_discharge", "float"),
    #     ("battery_charge_level", "float"),
    #     ("battery_charge_percentage", "float"),
    #     ("total_cost_after_solar", "float")
    # ])


    return pd.DataFrame(array, columns=[
        "date",
        "pv_total_power",
        "number_of_panels",
        "scaling_factor",
        "scaled_pv",
        "load_power",
        "grid_import_price",
        "grid_renewable_fraction",
        "cost_before_solar",
        "renewable_to_load",
        "supplied_pv",
        "battery_mode",
        "battery_charge_from_solar",
        "battery_charge_from_grid",
        "battery_charge",
        "battery_discharge_to_load",
        "battery_discharge_to_grid",
        "battery_discharge",
        "battery_charge_level",
        "battery_charge_percentage",
        "total_cost_after_solar"
    ])

def displayData(CustomerID):
    customer_data = getCustomerData(CustomerID)
    dates = np.sort(getDatesFromData(customer_data))
 
    print(f"Total Data Points:{len(dates)}")

    load = getAverageDayArray(CustomerID,5)
    total= getAverageDayArray(CustomerID,6)
    gridExport = getAverageDayArray(CustomerID,7)
    renew = getAverageDayArray(CustomerID,9)
    gridImportCost = getAverageDayArray(CustomerID, 8)

    print("Average Day PowerUsage")
    print(sum(load))
    print()
    print("Average Day")
    print(load)
    print()

    print("Total Average Day PV")
    print(sum(total))
    print()
    print("Average Day")
    print(total)
    print()

    print("Total Average Day GridExport")
    print(sum(gridExport))
    print()
    print("Average Day")
    print(gridExport)
    print()

    print("Average Day Renewable Fraction")
    print(sum(renew)/len(renew))
    print()
    print("Average Day")
    print(renew)
    print()

    print("Total Average Day Cost for Grid Import")
    print(sum(gridImportCost)/len(gridImportCost))
    print()
    print("Average Day")
    print(gridImportCost)
    print()

    totalGeneratedBonus = sum(total)-sum(load)

    profitFromGrid = 0
    if (totalGeneratedBonus > 0):
        profitFromGrid = totalGeneratedBonus * (sum(gridExport)/len(gridExport))
    elif (totalGeneratedBonus < 0):
        profitFromGrid = totalGeneratedBonus*(sum(gridImportCost)/len(gridImportCost))
    else:
        profitFromGrid = 0

    print("As an average, the cost for the day is " + str(np.int64(profitFromGrid)))
    
    numberOfSolarPanels = np.int64(getAverageDayArray(CustomerID,1)[0])
    scalabilityRatio = (numberOfSolarPanels-10)/numberOfSolarPanels

    gradient = np.empty(20)
    profitList = np.empty(20)
    optimumScalability = [10000,10000]

    for i in range (numberOfSolarPanels-10,numberOfSolarPanels+11):
        totalGeneratedBonus = (sum(total)*(scalabilityRatio*i))-sum(load)
        profitFromGrid = 0
        if (totalGeneratedBonus > 0):
            profitFromGrid = totalGeneratedBonus * (sum(gridExport*(scalabilityRatio*i))/len(gridExport))
        elif (totalGeneratedBonus < 0):
            profitFromGrid = totalGeneratedBonus*(sum(gridImportCost)/len(gridImportCost))
        else:
            profitFromGrid = 0

        profitList[i-numberOfSolarPanels-10] = profitFromGrid

        gradient = np.gradient(profitList)
        print("With ",str(i)," solar panels it would generate you on average this day: ",str(round(profitFromGrid,2)),"NZD")
        
        if (np.double(gradient[i-numberOfSolarPanels-10])<optimumScalability[0] and np.double(gradient[i-numberOfSolarPanels-10]) > 0):
            optimumScalability[0] = np.double(gradient[i-numberOfSolarPanels-10])
            optimumScalability[1] = i
        
        print(optimumScalability[0])

    print(np.int64(optimumScalability[1])," is the optimum amount of solar panels with no batteries") 


def batterySimulator(BatteryChargeRatio, numOfBatteries):
    batteryCapacity = BATTERY_CAPACITY*BatteryChargeRatio
    if (batteryChargePercentage(batteryCapacity)<=18):
        battery_mode = 2
    if (batteryChargePercentage(batteryCapacity) >= 90):
        battery_mode = 3
    
    batteryConsumption = 0

    if (battery_mode == 2):
        batteryConsumption = -1*numOfBatteries
    if (battery_mode == 3):
        batteryConsumption = 1*numOfBatteries

    return batteryConsumption

displayData(57)

# Website
customer_id = st.number_input('Customer ID', min_value=0, max_value=100, value=57, step=1)
no_of_panels = st.number_input('Number of panels', min_value=0, max_value=100, value=10, step=1)
no_of_batteries = st.number_input('Number of batteries', min_value=0, max_value=100, value=2, step=1)

df = generateModelTable(customer_id, no_of_batteries, no_of_panels)
start = st.date_input('Start date', min_value=min(df["date"]), max_value=max(df["date"]), value=min(df["date"]))
end = st.date_input('End date', min_value=min(df["date"]), max_value=max(df["date"]), value=max(df["date"]))

df = df[(df["date"] >= np.datetime64(start)) & (df["date"] <= np.datetime64(end))]
st.write(df)
st.line_chart(df, x="date", y=["load_power", "pv_total_power"])
st.line_chart(df, x="date", y=["battery_charge_level", "supplied_pv"])
