

---


# Степан Харитонов

## ML Engineer | Deep Learning | Scientific Computing | Mathematical Modeling

Фокус на разработке ML/AI систем, deep learning оптимизации и scientific computing. Опыт включает обучение нейросетей в условиях ограниченного вычислительного бюджета, медицинский анализ данных, промышленные ML-системы и вычислительное моделирование.

Интересы: deep learning optimization, ML systems, scientific computing, medical imaging, FEM, numerical methods, AI safety.

---

# Deep Learning Research Project — SMILE

## Zero-Order Optimization of ResNet18 under Compute Constraints

---

## Постановка задачи

Рассматривается обучение глубокой нейросети при ограничениях:

* отсутствуют аналитические градиенты;
* вычислительный бюджет строго ограничен (≤ 256 forward passes);
* требуется стабильное обучение ResNet18.

Формально задача оптимизации:

$$
\min_{\theta} \mathcal{L}(\theta)
$$

---

## Методология

### Zero-Order Optimization (SPSA)

Градиент аппроксимируется как:

$$
g(\theta) \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}
$$

Это позволяет выполнять обучение без backpropagation.

---

### Гибридная оптимизация

Используется комбинация:

* SPSA (исследование пространства параметров)
* Adam (стабилизация обновлений)

---

### Стратегия обучения

* curriculum learning (поэтапное усложнение данных)
* staged unfreezing слоёв
* стабилизация low-compute режима
* ограничение: ≤ 256 forward-pass операций

---

## Результаты

* Accuracy: **4.12% → 21.47%**
* стабильная сходимость при жестком ограничении вычислений
* улучшение обобщающей способности модели

---

## Технологии

Python · PyTorch · NumPy · Deep Learning · Optimization · Scientific Computing

Репозиторий:
[https://github.com/stepkhop/Solving-the-problem-on-SMILE](https://github.com/stepkhop/Solving-the-problem-on-SMILE)

---

# Medical Imaging & Scientific ML System

* кластеризация МРТ легких
* обработка медицинских изображений
* FEM mesh generation pipeline
* интеграция ML в scientific workflows
* подготовка данных для физиологического моделирования

Python · ML · Computer Vision · Clustering · FEM · Scientific Computing

[https://github.com/stepkhop/Reading_files](https://github.com/stepkhop/Reading_files)

---

# Industrial ML System (Oil & Gas Forecasting)

* анализ временных рядов промышленных данных
* feature engineering
* построение ML моделей прогнозирования
* внедрение в инженерные процессы

Python · Scikit-learn · Pandas · ML · Data Analysis

[https://disk.yandex.ru/i/7am2Gr5G_skPlQ](https://disk.yandex.ru/i/7am2Gr5G_skPlQ)
[https://disk.yandex.ru/d/0wAGhXKa8cGfxQ](https://disk.yandex.ru/d/0wAGhXKa8cGfxQ)
[https://disk.yandex.ru/d/Wd0v1GjjqenkBw](https://disk.yandex.ru/d/Wd0v1GjjqenkBw)

---

# Computer Vision / Biomedical Signal Processing

* анализ эхокардиографии
* обработка медицинских изображений
* временные биомедицинские сигналы
* feature extraction pipelines

[https://github.com/stepkhop/My-projects-commercial-and-non-commercial](https://github.com/stepkhop/My-projects-commercial-and-non-commercial)

---

# Data Structures & Algorithms

* бинарные деревья
* визуализация графов
* экспорт PNG
* алгоритмы обработки структур

[https://github.com/stepkhop/Developments-without-edits-](https://github.com/stepkhop/Developments-without-edits-)

---

# VR / Simulation Systems

* VR взаимодействия
* физические симуляции
* инженерные тренажеры
* Unity-based systems

[https://disk.yandex.ru/i/SQ_X1uPNmg2m3w](https://disk.yandex.ru/i/SQ_X1uPNmg2m3w)

---

# Skills

Python · C++
PyTorch · Scikit-learn · NumPy · Pandas
Deep Learning · Optimization · ML Systems
Mathematical Modeling · FEM · Numerical Methods
Git · Linux · Docker · WSL · Unity

---

# Selected Projects

[https://github.com/stepkhop/Solving-the-problem-on-SMILE](https://github.com/stepkhop/Solving-the-problem-on-SMILE)
[https://github.com/stepkhop/Reading_files](https://github.com/stepkhop/Reading_files)
[https://github.com/stepkhop/Developments-without-edits-](https://github.com/stepkhop/Developments-without-edits-)
[https://github.com/stepkhop/My-projects-commercial-and-non-commercial](https://github.com/stepkhop/My-projects-commercial-and-non-commercial)
[https://github.com/stepkhop/Refactoring_science_work](https://github.com/stepkhop/Refactoring_science_work)
[https://github.com/stepkhop/Temperature-converter-from-Fahrenheit-to-Celsius](https://github.com/stepkhop/Temperature-converter-from-Fahrenheit-to-Celsius)

---

# Contact

Telegram: [https://t.me/kra23z](https://t.me/kra23z)
GitHub: [https://github.com/stepkhop](https://github.com/stepkhop)
Email: [stepan2007p@mail.ru](mailto:stepan2007p@mail.ru)

---

---



# Stepan Kharitonov

## ML Engineer | Deep Learning | Scientific Computing | Mathematical Modeling

Focused on ML/AI systems, deep learning optimization, and scientific computing. Experience includes neural network training under strict compute constraints, medical data analysis, industrial ML systems, and computational modeling pipelines.

Interests: deep learning optimization, ML systems, scientific computing, medical imaging, FEM, numerical methods, AI safety.

---

# Deep Learning Research Project — SMILE

## Zero-Order Optimization of ResNet18 under Compute Constraints

---

## Problem Statement

We consider training a deep neural network under the following constraints:

* no access to analytical gradients;
* strict compute budget (≤ 256 forward passes);
* requirement for stable training of ResNet18.

Optimization objective:

$$
\min_{\theta} \mathcal{L}(\theta)
$$

---

## Methodology

### Zero-Order Optimization (SPSA)

Gradient is approximated as:

$$
g(\theta) \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}
$$

enabling gradient-free optimization.

---

### Hybrid Optimization Scheme

Combination of:

* SPSA (exploration of parameter space)
* Adam (stabilization of updates)

---

### Training Strategy

* curriculum learning
* staged layer unfreezing
* stabilization in low-compute regime
* constraint: ≤ 256 forward passes

---

## Results

* Accuracy: **4.12% → 21.47%**
* stable convergence under compute constraints
* improved generalization in low-resource settings

---

## Stack

Python · PyTorch · NumPy · Deep Learning · Optimization · Scientific Computing

Repository:
[https://github.com/stepkhop/Solving-the-problem-on-SMILE](https://github.com/stepkhop/Solving-the-problem-on-SMILE)

---

# Medical Imaging & Scientific ML System

* MRI lung clustering
* medical image preprocessing
* FEM mesh generation pipeline
* ML integration into scientific workflows
* preparation for physiological modeling

Python · ML · Computer Vision · Clustering · FEM · Scientific Computing

[https://github.com/stepkhop/Reading_files](https://github.com/stepkhop/Reading_files)

---

# Industrial ML System (Oil & Gas Forecasting)

* industrial time-series analysis
* feature engineering
* predictive ML models
* deployment into engineering workflows

Python · Scikit-learn · Pandas · ML · Data Analysis

[https://disk.yandex.ru/i/7am2Gr5G_skPlQ](https://disk.yandex.ru/i/7am2Gr5G_skPlQ)
[https://disk.yandex.ru/d/0wAGhXKa8cGfxQ](https://disk.yandex.ru/d/0wAGhXKa8cGfxQ)
[https://disk.yandex.ru/d/Wd0v1GjjqenkBw](https://disk.yandex.ru/d/Wd0v1GjjqenkBw)

---

# Computer Vision / Biomedical Signal Processing

* echocardiography analysis
* medical image processing
* biomedical time-series modeling
* feature extraction pipelines

[https://github.com/stepkhop/My-projects-commercial-and-non-commercial](https://github.com/stepkhop/My-projects-commercial-and-non-commercial)

---

# Data Structures & Algorithms

* binary tree generation
* graph visualization
* PNG export systems
* structural algorithm design

[https://github.com/stepkhop/Developments-without-edits-](https://github.com/stepkhop/Developments-without-edits-)

---

# VR / Simulation Systems

* VR interaction systems
* physics-based simulation
* industrial training environments
* Unity-based simulation tools

[https://disk.yandex.ru/i/SQ_X1uPNmg2m3w](https://disk.yandex.ru/i/SQ_X1uPNmg2m3w)

---

# Skills

Python · C++
PyTorch · Scikit-learn · NumPy · Pandas
Deep Learning · Optimization · ML Systems
Mathematical Modeling · FEM · Numerical Methods
Git · Linux · Docker · WSL · Unity

---

# Selected Projects

[https://github.com/stepkhop/Solving-the-problem-on-SMILE](https://github.com/stepkhop/Solving-the-problem-on-SMILE)
[https://github.com/stepkhop/Reading_files](https://github.com/stepkhop/Reading_files)
[https://github.com/stepkhop/Developments-without-edits-](https://github.com/stepkhop/Developments-without-edits-)
[https://github.com/stepkhop/My-projects-commercial-and-non-commercial](https://github.com/stepkhop/My-projects-commercial-and-non-commercial)
[https://github.com/stepkhop/Refactoring_science_work](https://github.com/stepkhop/Refactoring_science_work)
[https://github.com/stepkhop/Temperature-converter-from-Fahrenheit-to-Celsius](https://github.com/stepkhop/Temperature-converter-from-Fahrenheit-to-Celsius)

---

# Contact

Telegram: [https://t.me/kra23z](https://t.me/kra23z)
GitHub: [https://github.com/stepkhop](https://github.com/stepkhop)
Email: [stepan2007p@mail.ru](mailto:stepan2007p@mail.ru)

---
