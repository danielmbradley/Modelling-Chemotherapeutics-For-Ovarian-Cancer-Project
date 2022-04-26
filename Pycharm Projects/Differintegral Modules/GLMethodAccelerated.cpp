//
//  main.cpp
//  GLModel
//
//  Created by Daniel Bradley on 22/03/2022.
//

#include <iostream>
#include <math.h>
#include <vector>
#include <tuple>

using namespace std;

double dynamics_equation(double y){
    return -5 * exp(y);
}

double w_t_function(double w_t){
    return pow(w_t, 2)/0.001;
}

double w_t_alpha_function(double w_t_alpha){
    return pow(w_t_alpha, 2)/0.001;
}

double v_t_function(double v_t){
    return pow(v_t, 2)/0.1;
}

double alpha_function(double alpha){
    return -((0.1*log(alpha)) + (0.1*log(1-alpha)));
}

double x_0_function(double x_0){
    double toxic_concentration = 3.0;
    if (x_0 < toxic_concentration) {
        return x_0*1.5;
    } else {
        return 0;
    }
}

class CJA {

    public:
        double alpha;
        double binomial_coefficient;
        int i;

        CJA(double alpha_param) {
            alpha = alpha_param;
            binomial_coefficient = 1.0;
            i = 0;
        }

        double next(){
            double x = i;
            i++;
            if (x == 0) {
                return pow((-1.0), x) * binomial_coefficient;
            } else {
                binomial_coefficient = binomial_coefficient * ((alpha - (x-1))/((x-1) + 1));
                return pow((-1.0), x) * binomial_coefficient;
            }
        }

        double instant_coefficient_calculation(double alpha, int j){
            binomial_coefficient = 1.0;
            for (int i = 0; i < j; i++) {
                binomial_coefficient = binomial_coefficient * ((alpha - i) / (i + 1));
            }
            return pow((-1), j) * binomial_coefficient;
        }

};

double first_order_point_solution(double y, double gradient, double step){
    return y + (step * gradient);
}

double fractional_order_point_solution(double alpha, double step, int iteration, vector<double>& y_cache, double previous_solution){
    double summation = 0;
    CJA coefficient_generator = CJA(alpha);
    double _ = coefficient_generator.next();

    for (int i = iteration; i > 0; i--) {
        summation += coefficient_generator.next() * y_cache[i-1];
    }

    return pow(step, alpha) * dynamics_equation(previous_solution) - summation;
}

double provided_dynamics_fractional_order_point_solution(double alpha, double step, int iteration, vector<double>& y_cache, double previous_solution, double dynamics){
    double summation = 0;
    CJA coefficient_generator = CJA(alpha);
    double _ = coefficient_generator.next();

    for (int i = iteration; i > 0; i--) {
        summation += coefficient_generator.next() * y_cache[i-1];
    }

    return pow(step, alpha) * dynamics - summation;
}

tuple<vector<double>, vector<vector<double>>> fractional_order_solution_to_first_order(double alpha, double step, int number_of_iterations, vector<double>& initial_conditions){

    if (initial_conditions.size() != 1) {
        cout <<  "Please check the parameters: number of initial conditions and math.ceil(order) should match but do not";
    }

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    y_values.push_back(initial_conditions);

    for (int x = 1; x < number_of_iterations; x++) {
        x_values.push_back(x_values[x - 1] + step);
        y_values[0].push_back(fractional_order_point_solution(alpha, step, x, y_values[0], y_values[0][x - 1]));
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> fractional_order_solution_from_first_to_nth_order(double alpha, double step, int number_of_iterations, vector<double> initial_conditions){

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    vector<double> condition;

    for (int x = 0; x < initial_conditions.size(); x++) {
        condition.push_back(initial_conditions[x]);
        y_values.push_back(condition);
        condition.erase(condition.begin());
    }

    for (int x = 1; x < number_of_iterations; x++) {
        x_values.push_back(x_values[x - 1] + step);
        for (int y = 0; y < ceil(alpha); y++) {
            if (y == 0) {
                y_values[y].push_back(fractional_order_point_solution(1.0, step, x, y_values[y], y_values[ceil(alpha) - 1][x-1]));
            } else if ((floor(alpha) == alpha and y > 0) or (floor(alpha) != alpha and y < floor(alpha))){
                y_values[y].push_back(provided_dynamics_fractional_order_point_solution(1.0, step, x, y_values[y], y_values[ceil(alpha) - 1][x-1], y_values[y - 1][x - 1]));
            } else {
                double current_system_alpha = alpha - floor(alpha);
                y_values[y].push_back(provided_dynamics_fractional_order_point_solution(current_system_alpha, step, x, y_values[y], y_values[ceil(alpha) - 1][x-1], y_values[y - 1][x - 1]));
            }
        }
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> fractional_order_solution(double alpha, double integration_step, double ending_point, vector<double> initial_conditions) {

    if (initial_conditions.size() != ceil(alpha)) {
        cout <<  "Please check the parameters: number of initial conditions and ceil(order) should match but do not";
    }

    if (alpha <= 0) {
        cout <<  "Please check the parameters: alpha must be a positive value";
    }

    double number_of_iterations = ceil(ending_point / integration_step);

    tuple<vector<double>, vector<vector<double>>> results;

    vector<vector<double>> y_values;

    if (alpha < 1) {
        results = fractional_order_solution_to_first_order(alpha, integration_step, number_of_iterations, initial_conditions);
    } else if (alpha >= 1) {
        results = fractional_order_solution_from_first_to_nth_order(alpha, integration_step, number_of_iterations, initial_conditions);
    }
    return results;
}

tuple<vector<double>, vector<vector<double>>> accelerated_fractional_order_solution_to_first_order(double alpha, double step, int number_of_iterations, vector<double>& initial_conditions){

    if (initial_conditions.size() != 1) {
        cout <<  "Please check the parameters: number of initial conditions and math.ceil(order) should match but do not";
    }

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    y_values.push_back(initial_conditions);

    if (alpha == 1) {
        for (int x = 1; x < number_of_iterations; x++) {
            x_values.push_back(x_values[x - 1] + step);
            y_values[0].push_back(first_order_point_solution(y_values[0][x-1], dynamics_equation(y_values[0][x-1]), step));
        }
    } else if (alpha < 1) {
        for (int x = 1; x < number_of_iterations; x++) {
            x_values.push_back(x_values[x - 1] + step);
            y_values[0].push_back(fractional_order_point_solution(alpha, step, x, y_values[0], y_values[0][x - 1]));
        }
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> accelerated_fractional_order_solution_from_first_to_nth_order(double alpha, double step, int number_of_iterations, vector<double> initial_conditions){

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    vector<double> condition;

    for (int x = 0; x < initial_conditions.size(); x++) {
        condition.push_back(initial_conditions[x]);
        y_values.push_back(condition);
        condition.erase(condition.begin());
    }

    for (int x = 1; x < number_of_iterations; x++) {
        x_values.push_back(x_values[x - 1] + step);
        for (int y = 0; y < ceil(alpha); y++) {
            if (y == 0) {
                y_values[y].push_back(first_order_point_solution(y_values[0][x-1], dynamics_equation(y_values[0][x-1]), step));
            } else if ((floor(alpha) == alpha and y > 0) or (floor(alpha) != alpha and y < floor(alpha))){
                y_values[y].push_back(first_order_point_solution(y_values[0][x-1], y_values[y - 1][x - 1], step));
            } else {
                double current_system_alpha = alpha - floor(alpha);
                y_values[y].push_back(provided_dynamics_fractional_order_point_solution(current_system_alpha, step, x, y_values[y], y_values[ceil(alpha) - 1][x-1], y_values[y - 1][x - 1]));
            }
        }
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> accelerated_fractional_order_solution(double alpha, double integration_step, double ending_point, vector<double> initial_conditions) {

    if (initial_conditions.size() != ceil(alpha)) {
        cout <<  "Please check the parameters: number of initial conditions and ceil(order) should match but do not";
    }

    if (alpha <= 0) {
        cout <<  "Please check the parameters: alpha must be a positive value";
    }

    double number_of_iterations = ceil(ending_point / integration_step);

    tuple<vector<double>, vector<vector<double>>> results;

    vector<vector<double>> y_values;

    if (alpha < 1) {
        results = accelerated_fractional_order_solution_to_first_order(alpha, integration_step, number_of_iterations, initial_conditions);
    } else if (alpha >= 1) {
        results = accelerated_fractional_order_solution_from_first_to_nth_order(alpha, integration_step, number_of_iterations, initial_conditions);
    }
    return results;
}

tuple<vector<double>, vector<vector<double>>> optimise_accelerated_fractional_order_solution_to_first_order(double alpha, double step, int number_of_iterations, vector<double>& initial_conditions){

    if (initial_conditions.size() != 1) {
        cout <<  "Please check the parameters: number of initial conditions and math.ceil(order) should match but do not";
    }

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    y_values.push_back(initial_conditions);

    if (alpha == 1) {
        for (int x = 1; x < number_of_iterations; x++) {
            x_values.push_back(x_values[x - 1] + step);
            y_values[0].push_back(first_order_point_solution(y_values[0][x-1], dynamics_equation(y_values[0][x-1]), step));
        }
    } else if (alpha < 1) {
        for (int x = 1; x < number_of_iterations; x++) {
            x_values.push_back(x_values[x - 1] + step);
            y_values[0].push_back(fractional_order_point_solution(alpha, step, x, y_values[0], y_values[0][x - 1]));
        }
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> optimise_accelerated_fractional_order_solution_from_first_to_nth_order(double alpha, double step, int number_of_iterations, vector<double> initial_conditions){

    vector<double> x_values = {0};

    vector<vector<double>> y_values;

    vector<double> condition;

    for (int x = 0; x < initial_conditions.size(); x++) {
        condition.push_back(initial_conditions[x]);
        y_values.push_back(condition);
        condition.erase(condition.begin());
    }

    for (int x = 1; x < number_of_iterations; x++) {
        x_values.push_back(x_values[x - 1] + step);
        for (int y = 0; y < ceil(alpha); y++) {
            if (y == 0) {
                y_values[y].push_back(first_order_point_solution(y_values[y][x-1], dynamics_equation(y_values[ceil(alpha)-1][x-1]), step));
            } else if ((floor(alpha) == alpha and y > 0) or (floor(alpha) != alpha and y < floor(alpha))){
                y_values[y].push_back(first_order_point_solution(y_values[y][x-1], y_values[y - 1][x - 1], step));
            } else {
                double current_system_alpha = alpha - floor(alpha);
                y_values[y].push_back(provided_dynamics_fractional_order_point_solution(current_system_alpha, step, x, y_values[y], y_values[ceil(alpha) - 1][x-1], y_values[y - 1][x - 1]));
            }
        }
    }
    return tuple<vector<double>, vector<vector<double>>>{x_values, y_values};
}

tuple<vector<double>, vector<vector<double>>> optimise_accelerated_fractional_order_solution(vector<double> parameters, double integration_step, double ending_point) {

    double alpha = parameters[0];

    vector<double> initial_conditions;

    for (int initial_condition = 1; initial_condition < parameters.size(); initial_condition++) {
        initial_conditions.push_back(parameters[initial_condition]);
    }

    if (initial_conditions.size() != ceil(alpha)) {
        cout <<  "Please check the parameters: number of initial conditions and ceil(order) should match but do not";
    }

    if (alpha <= 0) {
        cout <<  "Please check the parameters: alpha must be a positive value";
    }

    double number_of_iterations = ceil(ending_point / integration_step);

    tuple<vector<double>, vector<vector<double>>> results;

    vector<vector<double>> y_values;

    if (alpha < 1) {
        results = optimise_accelerated_fractional_order_solution_to_first_order(alpha, integration_step, number_of_iterations, initial_conditions);
    } else if (alpha >= 1) {
        results = optimise_accelerated_fractional_order_solution_from_first_to_nth_order(alpha, integration_step, number_of_iterations, initial_conditions);
    }
    return results;
}

extern "C" void fractional_order_dynamics_solution(double * parameters,
                       int length_of_parameters,
                       double integration_step,
                       double ending_point,
                       double * results) {

    vector<double> params;
    tuple<vector<double>, vector<vector<double>>> unparsed_results;

    for (int i = 0; i < length_of_parameters; i++) {
        params.push_back(parameters[i]);
    }

    unparsed_results = optimise_accelerated_fractional_order_solution(params, integration_step, ending_point);

    for (int i = 0; i < get<0>(unparsed_results).size(); i++) {
        results[i] = get<0>(unparsed_results)[i];
    }

    for (int i = 0; i < get<1>(unparsed_results)[0].size(); i++) {
        results[i+get<0>(unparsed_results).size()] = get<1>(unparsed_results)[get<1>(unparsed_results).size()-1][i];
    }
}

extern "C" double cost(double * parameters,
                       int length_of_parameters,
                       double * y_dataset,
                       double * x_dataset,
                       int length_of_dataset,
                       double integration_step,
                       double ending_point) {

    vector<double> params;
    vector<double> w_t;
    vector<double> w_t_alpha;
    vector<double> v_t;
    vector<double> valid_indexs;
    double cost;
    double summed = 0;
    double initial_x = 0;
    tuple<vector<double>, vector<vector<double>>> results;
    vector<double> solution;

    if (length_of_dataset != ceil(ending_point / integration_step)) {
        for (int i = 0; i < ceil(ending_point / integration_step); i++) {
            if (x_dataset[i] == initial_x) {
                valid_indexs.push_back(i);
            }
            initial_x += integration_step;
        }
    }

    for (int i = 0; i < length_of_parameters; i++) {
        params.push_back(parameters[i]);
    }

    for (int i = length_of_parameters; i < length_of_dataset+length_of_parameters; i++) {
        w_t.push_back(parameters[i]);
    }

    for (int i = length_of_parameters+length_of_dataset; i < (2*length_of_dataset)+length_of_parameters; i++) {
        w_t_alpha.push_back(parameters[i]);
    }

    cost = alpha_function(parameters[0]) + x_0_function(parameters[1]);

    results = optimise_accelerated_fractional_order_solution(params, integration_step, ending_point);

    solution = get<1>(results)[0];

    if (length_of_dataset == ceil(ending_point / integration_step)){
        for (int i = 0; i < length_of_dataset; i++) {
            cost += w_t_function(w_t[i]) + w_t_alpha_function(w_t_alpha[i]) + v_t_function(y_dataset[i]-(solution[i]+w_t[i]));
        }
    }

    if (length_of_dataset != ceil(ending_point / integration_step)){
        for (int i = 0; i < valid_indexs.size(); i++) {
            cost += w_t_function(w_t[valid_indexs[i]]) + w_t_alpha_function(w_t_alpha[valid_indexs[i]]) + v_t_function(y_dataset[i]-(solution[valid_indexs[i]]+w_t[valid_indexs[i]]));
        }
    }
    return cost;
}