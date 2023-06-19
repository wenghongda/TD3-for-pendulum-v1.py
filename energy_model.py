from environment import Environment
import numpy as np
from typing import List,Any,Iterable,Union
class Device(Environment):
    def __init__(self,efficiency:float):
        """Initialize Device, every device should specify its own efficiency.

        Parameters
        ----------
        efficiency (float): the running efficiency of devices considering energy consumption
        """
        super().__init__()
        self.efficiency = efficiency

    @property
    def efficiency(self) -> float:
        """Technical efficiency."""

        return self.__efficiency

    @efficiency.setter
    def efficiency(self,efficiency:float):
        if efficiency is None:
            self.__efficiency = 1.0
        else:
            assert efficiency > 0 and efficiency < 1.0
            self.__efficiency = efficiency

class ElectricDevice(Device):
    """Base electric device class like electricity load in resident building(HVACs) which can't generate energy but consume energy.

    Parameters
    ----------
    nominal_power (kw):float
        Electric device nominal power >= 0. If == 0,it should be set to 0.00001 to avoid 'ZeroDivisionError'.
    """
    def __init__(self,nominal_power:float):
        super().__init__()
        self.nominal_power = nominal_power

    @property
    def nominal_power(self) -> float:
        """Nominal power."""

        return self.__nominal_power
    @property
    def electricity_consumption(self) -> List[float]:
        """Electricity consumption time series."""

        return self.__electricity_consumption
    @property
    def available_nominal_power(self) -> float:
        """Difference between 'nominal_power' and 'electricity_consumption' at current 'time_step'"""

        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]

    @nominal_power.setter
    def nominal_power(self,nominal_power:float):
        if nominal_power is None or nominal_power == 0:
            self.__nominal_power = ZeroDivisionError
        else:
            assert nominal_power >= 0
            self.__nominal_power = nominal_power

    def update_electricity_consumption(self,electricity_consumption:float):
        """Updates 'electricity_consumption' at current 'time_step'.

        Parameters
        ----------
        electricity_consumption:float
            value that needs to be added to list 'electricity_consumption' at current 'time_step' and must > 0
        """

        assert electricity_consumption > 0
        self.__electricity_consumption[self.time_step] += electricity_consumption

    def next_time_step(self):
        """Advance to the next ’time_step‘ """
        super().next_time_step()
        self.__electricity_consumption.append(0.0)#ready for the next_time_step 's calculation of electricity consumption

    def reset(self):
        super().reset()
        self.__electricity_consumption = [0.0]
class StorageDevice(Device):
    def __init__(self,capacity:float,efficiency:float = None,loss_efficiency:float=None,initial_soc:float=None):
        self.__capacity = capacity
        self.__loss_coefficiency = loss_efficiency
        self.__initial_soc = initial_soc
        super().__init__(efficiency=efficiency)

    @property
    def capacity(self) -> float:
        """Maximum amount of energy that the storage device can store [KWH]"""

        return self.__capacity
    @property
    def loss_coefficiency(self) -> float:

        return self.__loss_coefficiency
    @property
    def initial_soc(self) -> float:

        return self.__initial_soc
    @property
    def soc(self) -> List[float]:
        """State of charge time series (%)"""
        return self.__soc
    @property
    def soc_init(self) -> float:
        """Latest state of charge after considerating standby hourly losses."""

        return self.__soc[-1] * 1 # `1` means no losses
    @property
    def energy_balance(self) -> List[float]:
        """Charge/Discharged energy in time series[KWH]."""

        return self.__energy_balance
    @property
    def round_trip_efficiency(self) -> float:
        """Efficiency square root."""

        return self.efficiency ** 0.5

    @capacity.setter
    def capacity(self,capacity:float):
        if capacity is None or capacity == 0:
            self.__capacity = ZeroDivisionError
        else:
            assert capacity >= 0
            self.__capacity = capacity
    @loss_coefficiency.setter
    def loss_coefficiency(self,loss_coefficient:float):
        if loss_coefficient is None:
            self.__loss_coefficiency = 0.006
        else:
            assert  0 <= loss_coefficient <= 1
            self.__loss_coefficiency = loss_coefficient
    @initial_soc.setter
    def initial_soc(self,initial_soc:float):
        if initial_soc is None:
            self.__initial_soc = 0
        else:
            assert  0 <= initial_soc <= self.capacity
            self.__initial_soc = initial_soc

    def charge(self,energy:float):
        """Charges or Discharges behaviors that change the soc state of storage device.

        Parameters
        ----------
        energy:float
            Energy to charge if (+) else to discharge(-) in [KWH].
        ----------

        Notes
        ----------
        If charging, soc = min('soc_init' + energy * 'round_trip_efficiency',capacity)
        If discharging, soc = max('soc_init' + energy/'round_trip_efficiency',0)
        """
        soc = min(self.soc_init + energy * self.seconds_per_time_step * self.round_trip_efficiency/3600,self.capacity) if energy > 0 + \
        else max(0,self.soc_init + energy * self.seconds_per_time_step/(self.round_trip_efficiency * 3600))
        self.__soc.append(soc)
        self.__energy_balance.append(self.set_energy_balance())
    def set_energy_balance(self):
        """Calculating energy balance.
        The energy balance is a derived quantity and is the product or quotient of the difference between consecutive SOCs and `round_trip_efficiency`
        for discharge or charge events respectively thus, thus accounts for energy losses to environment during charging and discharge.
        """
        # actual energy charged/discharged irrespective of what is determined in the step function after
        # taking into account storage design limits e.g. maximum power input/output, capacity
        previous_soc = self.initial_soc if self.time_step == 0 else self.soc[-2]
        energy_balance = self.soc[-1] - previous_soc * (1.0 - self.loss_coefficient)
        energy_balance = energy_balance / self.round_trip_efficiency if energy_balance >= 0 else energy_balance * self.round_trip_efficiency

        return energy_balance
    def reset(self):
        """Reset 'StorageDevice' to initial state."""

        self.__soc = [self.initial_soc]
        self.__energy_balance = [0.0]

class Hydrogen_tank(StorageDevice):
    r"""Base hydrogen gas storage class.

        Parameters
        ----------
        capacity :float
            Maximum amount of hydrogen gas the storage device can store in [Mpa].Must be >= 0 and if ==0 or None,set to 0.00001 to avoide `ZeroDivisionError` .
        volume : flaot
            the volume of the hydrogen tank(L)
        temperature :float
            the ambient temperature of the hydrogen tank[K]. Must be > 0 if None,default to be 313K.
    """
    def __init__(self,current_pressure:float,capacity:float,volume:float,temperature:float = None):
        super().__init__(capacity=capacity)
        self.__soc_history = [current_pressure/capacity]
        self.volume = volume
        self.temperature = 313# default to be 313.15K

    @property
    def soc_history(self):

        return self.__soc_history

    def charge(self,hydrogen_mass_el:float,hydrogen_mass_fc:float,hydrogen_mass_fcv:float = None,b:float = 7.691*10**(-3)):
        """Charges or discharges storage with respect to specified energy while considering `capacity` and `soc_init` limitations and, energy losses to the environment quantified by `efficiency`.

        Parameters
        ----------
        hydrogen_mass_el (kg) float:
            the hydrogen mass that electrolyzer generates at current time_step
        hydrogen_mass_fc (kg) float:
            the hydrogen mass that fuel cell consumes at current time_step
        hydrogen_mass_fcv (kg) float:
            the hydrogen mass that fuel cell vehicle consumes at current time_step
        b (1) float:
            a constant number in Abel-Nobel state equation to calculate hydrogen property in high pressure circumstances
            and default to be 7.691e-3
        R (1) float:
            a constant number in Abel-Nobel state equation to calculation hydrogen property in high pressure circumstances
            and default to be 4124J/(Kg*K)
        T (K) int:constant temperature 313K
        """
        soc = self.soc_history[-1] + (4124 * 313 * (
                    1 / (self.volume / (hydrogen_mass_el) - b) - 1 / (self.volume / (hydrogen_mass_fc) - b) - 1 / (
                        self.volume / (hydrogen_mass_fcv) - b))) / self.capacity
        self.__soc_history.append(soc)

        def get_max_input_power(self) -> float:

            pass
        def get_max_output_power(self) -> float:
            """


            """
            pass


class PV(ElectricDevice):
    r"""Base photovoltaic array class.

    Parameters
    ----------
    nominal_power :float
        PV array output power in [kw]. Must be >= 0
    Other Parameters
    ----------------
    **kwargs:Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, nominal_power: float, **kwargs: Any):
        super().__init__(nominal_power=nominal_power, **kwargs)

    def get_generation(self, inverter_ac_power_per_kw: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        r"""Get solar generation output.
        Parameters
        ----------
        inverter_ac_power_perk-w:Union[float,Iterable[float]]
            Inverter AC power output per kW of PV capcaity in [W/kW].
        Returns
        -------
        generation : Union[float,Iterable[float]]
            Solar generation as a single value or time series depending on input parameter types.

        """
        return self.nominal_power * np.array(inverter_ac_power_per_kw) / 1000

    def autosize(self, demand: Iterable[float], safety_factor: float = None):
        safety_factor = 1.0 if safety_factor is None else safety_factor
        self.nominal_power = np.nanmax(np.array(demand) / self.efficiency) * safety_factor


class Electrolyzer(ElectricDevice):
    r"""the electrolyzer is used to consume water and electricity to produce hydrogen
    Parameters
    ----------
    maximum_power:float
        The maximum running power of Electrolyzer
    minimum_power:float
        The minimum running power of electrolyzer
    """

    def __init__(self, nominal_power: float, efficiency: float, initial_output_power: float):
        super().__init__(nominal_power)
        self.__initial_output_power = initial_output_power
        self.__output_power_history = [initial_output_power]
        self.__output_power = initial_output_power

    @Device.efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.8 if efficiency is None else efficiency
        ElectricDevice.efficiency.fset(self, efficiency)

    @property
    def output_power_history(self) -> list[float]:
        return self.__output_power_history

    @property
    def output_power(self) -> float:
        return self.__output_power_history[-1]

    @output_power.setter
    def output_power(self, output_power) -> float:
        self.__output_power = output_power
        self.__output_power_history.append(output_power)

    def get_max_output_power(self, max_electric_power: Union):
        if max_electric_power is None:
            return self.available_nominal_power * self.efficiency
        else:
            return np.min(max_electric_power, self.available_nominal_power) * self.efficiency

    def get_hydrogen_generation(self):
        """
        n_el = eta_el * P_el /(LHV_H2)
        eta_el = efficiency
        P_el = running_power(KW)
        LHV_H2 = 240(MJ/Kmol)
        self.hydrogen_generation is the produced hydrogen mole flow rate(mol/s)
        2 * self.hydrogen_generation is the produced hydrogen mass flow rate[g/s]
        self.hydrogen_generation * 2 * (self.seconds_per_time_step) = g
        """

        return np.array(self.output_power_history, dtype=np.float64) * self.efficiency / 240 * 2 * (
            self.seconds_per_time_step)

    def reset(self):
        """Reset `Electrolyzer` to initial state."""
        super().reset()
        self.__output_power_history = []


class Fuel_cell(ElectricDevice):
    def __init__(self, nominal_power: float, efficiency: float, initial_output_power: float):
        super().__init__(nominal_power=nominal_power, efficiency=efficiency)
        self.__output_power = initial_output_power
        self.__output_power_history = [initial_output_power]

    @property
    def output_power(self):
        return self.__output_power

    @property
    def output_power_history(self):
        return self.__output_power_history

    @Device.efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.8 if efficiency is None else efficiency
        ElectricDevice.efficiency.fset(self, efficiency)

    @property
    def output_power_history(self) -> list[float]:
        return self.__output_power_history

    @property
    def output_power(self) -> float:
        return self.__output_power_history[-1]

    @output_power.setter
    def output_power(self, output_power) -> float:
        self.__output_power = output_power
        self.__output_power_history.append(output_power)

    def get_max_output_power(self, max_electric_power: Union):
        if max_electric_power is None:
            return self.available_nominal_power * self.efficiency
        else:
            return np.min(max_electric_power, self.available_nominal_power) * self.efficiency

    def get_hydrogen_consumption(self):
        """
        n_fc = P_fc /(eta_fc * LHV_H2)
        eta_el = efficiency
        P_fc = running_power(KW)
        LHV_H2 = 240(MJ/Kmol)
        self.hydrogen_consumption is the consumed hydrogen mole flow rate(mol/s)
        2 * self.hydrogen_generation is the consumend hydrogen mass flow rate[g/s]
        self.hydrogen_generation * 2 * (self.seconds_per_time_step) = g / seconds_per_time_step
        """
        return np.array(self.output_power_history, dtype=np.float64) / (self.efficiency * 240) * 2 * (
            self.seconds_per_time_step)


class Battery(ElectricDevice, StorageDevice):
    r"""Base electricity storage class.

    Parameters
    ----------
    capacity : float
        Maximum amount of energy the storage device can store in [kWh]. Must be >= 0 and if == 0 or None, set to 0.00001 to avoid `ZeroDivisionError`.
    nominal_power: float
        Maximum amount of electric power that the battery can use to charge or discharge.
    capacity_loss_coefficient : float, default: 0.00001
        Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity).
    power_efficiency_curve: list, default: [[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]]
        Charging/Discharging efficiency as a function of the power released or consumed.
    capacity_power_curve: list, default: [[0.0, 1],[0.8, 1],[1.0, 0.2]]
        Maximum power of the battery as a function of its current state of charge.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super classes.
    """

    def __init__(self, capacity: float, nominal_power: float, capacity_loss_coefficient: float = None,
                 power_efficiency_curve: list[list[float]] = None, capacity_power_curve: list[list[float]] = None,
                 **kwargs: any):
        self.__efficiency_history = []
        self.__capacity_history = []
        super().__init__(capacity=capacity, nominal_power=nominal_power, **kwargs)
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.power_efficiency_curve = power_efficiency_curve
        self.capacity_power_curve = capacity_power_curve

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` degradation and `soc_init` limitations, losses to the environment quantified by `efficiency`, `power_efficiency_curve` and `capacity_power_curve`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].

        Notes
        -----
        If charging, soc = min(`soc_init` + energy*`efficiency`, `max_input_power`, `capacity`)
        If discharging, soc = max(0, `soc_init` + energy/`efficiency`, `max_output_power`)
        """

        energy = min(energy, self.get_max_input_power()) if energy >= 0 else max(-self.get_max_output_power(), energy)
        self.efficiency = self.get_current_efficiency(energy)
        super().charge(energy)

    def get_max_output_power(self) -> float:
        r"""Get maximum output power while considering `capacity_power_curve` limitations if defined otherwise, returns `nominal_power`.

        Returns
        -------
        max_output_power : float
            Maximum amount of power that the storage unit can output [kW].
        """

        return self.get_max_input_power()

    def get_max_input_power(self) -> float:
        r"""Get maximum input power while considering `capacity_power_curve` limitations if defined otherwise, returns `nominal_power`.

        Returns
        -------
        max_input_power : float
            Maximum amount of power that the storage unit can use to charge [kW].
        """
        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        # if self.capacity_power_curve is not None:
        #  capacity = self.capacity_history[-2] if len(self.capacity_history) > 1 else self.capacity
        max_input_power = self.nominal_power
        return max_input_power

    def get_current_efficiency(self, energy: float) -> float:
        efficiency = self.efficiency
        return efficiency

    @StorageDevice.capacity.getter
    def capacity(self) -> float:
        r"""Current time step maximum amount of energy the storage device can store in [kWh]"""

        return self.capacity_history[-1]

    @StorageDevice.efficiency.getter
    def efficiency(self) -> float:
        """Current time step technical efficiency."""

        return self.efficiency_history[-1]

    @ElectricDevice.electricity_consumption.getter
    def electricity_consumption(self) -> list[float]:
        r"""Electricity consumption time series."""

        return self.energy_balance

    @property
    def capacity_loss_coefficient(self) -> float:
        """Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity)."""

        return self.__capacity_loss_coefficient

    @property
    def power_efficiency_curve(self) -> np.ndarray:
        """Charging/Discharging efficiency as a function of the power released or consumed."""

        return self.__power_efficiency_curve

    @property
    def capacity_power_curve(self) -> np.ndarray:
        """Maximum power of the battery as a function of its current state of charge."""

        return self.__capacity_power_curve

    @property
    def efficiency_history(self) -> list[float]:
        """Time series of technical efficiency."""

        return self.__efficiency_history

    @property
    def capacityMcity_history(self) -> list[float]:
        """Time series of maximum amount of energy the storage device can store in [kWh]."""

        return self.__capacity_history

    @capacity.setter
    def capacity(self, capacity: float):
        capacity = ZERO_DIVISION_CAPACITY if capacity is None or capacity == 0 else capacity
        StorageDevice.capacity.fset(self, capacity)
        self.__capacity_history.append(capacity)

    @efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.9 if efficiency is None else efficiency
        StorageDevice.efficiency.fset(self, efficiency)
        self.__efficiency_history.append(efficiency)

    @capacity_loss_coefficient.setter
    def capacity_loss_coefficient(self, capacity_loss_coefficient: float):
        if capacity_loss_coefficient is None:
            capacity_loss_coefficient = 1e-5
        else:
            pass

        self.__capacity_loss_coefficient = capacity_loss_coefficient

    @power_efficiency_curve.setter
    def power_efficiency_curve(self, power_efficiency_curve: list[list[float]]):
        if power_efficiency_curve is None:
            power_efficiency_curve = [[0, 0.83], [0.3, 0.83], [0.7, 0.9], [0.8, 0.9], [1, 0.85]]
        else:
            pass

        self.__power_efficiency_curve = np.array(power_efficiency_curve).T

    @capacity_power_curve.setter
    def capacity_power_curve(self, capacity_power_curve: list[list[float]]):
        if capacity_power_curve is None:
            capacity_power_curve = [[0.0, 1], [0.8, 1], [1.0, 0.2]]
        else:
            pass

        self.__capacity_power_curve = np.array(capacity_power_curve).T

    @property
    def efficiency_history(self) -> list[float]:
        """Time series of technical efficiency."""

        return self.__efficiency_history

    @property
    def capacity_history(self) -> list[float]:
        """Time series of maximum amount of energy the storage device can store in [kWh]."""

        return self.__capacity_history

    @efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.9 if efficiency is None else efficiency
        StorageDevice.efficiency.fset(self, efficiency)
        self.__efficiency_history.append(efficiency)

    @capacity_loss_coefficient.setter
    def capacity_loss_coefficient(self, capacity_loss_coefficient: float):
        if capacity_loss_coefficient is None:
            capacity_loss_coefficient = 1e-5
        else:
            pass

        self.__capacity_loss_coefficient = capacity_loss_coefficient

    @power_efficiency_curve.setter
    def power_efficiency_curve(self, power_efficiency_curve: list[list[float]]):
        if power_efficiency_curve is None:
            power_efficiency_curve = [[0, 0.83], [0.3, 0.83], [0.7, 0.9], [0.8, 0.9], [1, 0.85]]
        else:
            pass

        self.__power_efficiency_curve = np.array(power_efficiency_curve).T

    @capacity_power_curve.setter
    def capacity_power_curve(self, capacity_power_curve: list[list[float]]):
        if capacity_power_curve is None:
            capacity_power_curve = [[0.0, 1], [0.8, 1], [1.0, 0.2]]
        else:
            pass

        self.__capacity_power_curve = np.array(capacity_power_curve).T

    def degrade(self) -> float:
        r"""Get amount of capacity degradation.

        Returns
        -------
        capacity:float
            Maximum amount of energy the storage device can store in [kWh].
        Notes
        ------
        degradation = `capacity_loss_coef` * `capacity_history[0]` * abs(`energy_balance[-1]`)/(2*`capacity`)
        """
        # Calculating the degradation of the battery: new max. capacity of the battery after c

    def reset(self):
        r"""Reset `Battery` to initial state."""
        super().reset()
        self.__efficiency_history = self.__efficiency_history[0:1]
        self.__capacity_history = self.__capacity_history[0:1]