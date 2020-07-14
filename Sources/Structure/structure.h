#ifndef STRUCTURE_H
#define STRUCTURE_H

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

class LocalEnvironment;

// Structure class.
class Structure {
  friend class LocalEnvironment;

  Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse, cell_dot,
      cell_dot_inverse, positions, wrapped_positions;

public:
  Structure();

  Structure(const Eigen::MatrixXd &cell, const std::vector<int> &species,
            const Eigen::MatrixXd &positions,
            const std::unordered_map<int, double> &mass_dict =
                std::unordered_map<int, double>{},
            const Eigen::MatrixXd &prev_positions = Eigen::MatrixXd::Zero(0, 3),
            const std::vector<std::string> &species_labels =
                std::vector<std::string>{});

  Structure(const Eigen::MatrixXd &cell,
            const std::vector<std::string> &species,
            const Eigen::MatrixXd &positions,
            const std::unordered_map<std::string, double> &mass_dict =
                std::unordered_map<std::string, double>{},
            const Eigen::MatrixXd &prev_positions = Eigen::MatrixXd::Zero(0, 3),
            const std::vector<std::string> &species_labels =
                std::vector<std::string>{});

  void set_structure(
      const Eigen::MatrixXd &cell, const std::vector<int> &species,
      const Eigen::MatrixXd &positions,
      const std::unordered_map<int, double> &mass_dict =
          std::unordered_map<int, double>{},
      const Eigen::MatrixXd &prev_positions = Eigen::MatrixXd::Zero(0, 3),
      const std::vector<std::string> &species_labels =
          std::vector<std::string>{});

  // Cell setter and getter.
  void set_cell(const Eigen::MatrixXd &cell);
  const Eigen::MatrixXd &get_cell() const;

  // Position setter and getter.
  void set_positions(const Eigen::MatrixXd &positions);
  const Eigen::MatrixXd &get_positions() const;
  const Eigen::MatrixXd &get_wrapped_positions();

  Eigen::MatrixXd wrap_positions();
  double get_max_cutoff();

  std::vector<int> coded_species;
  std::vector<std::string> species_labels;
  std::unordered_map<int, double> mass_dict;
  Eigen::MatrixXd prev_positions, forces, stds, local_energies,
      local_energy_stds, partial_stresses, partial_stress_stds, stress,
      stress_stds;
  double max_cutoff, volume, potential_energy;
  int nat;
};

#endif

