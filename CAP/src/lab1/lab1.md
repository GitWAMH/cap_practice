# Práctica 1

## Antes de nada 
* ¿Cómo saber el equipo disponible?
    * Modelo del procesador
    * Número de cores
    * Vectorización: *sse*, *avx*...
    * Especificaciones a consultar en la página de [Intel](https://ark.intel.com/content/www/es/es/ark.html\#@PanelLabel122139)

```console
user@lab:~ $  more /proc/cpuinfo 
...
processor	: 3
...
model name	: Intel(R) Xeon(R) CPU E3-1225 v3 @ 3.20GHz
...
flags		: ... avx ... avx2 ...
....
```

## Opciones de Compilación
* Opciones del compilador para llevar a cabo optimizaciones

## Autovectorización
* La opción **-O2** del compilador de [Intel® C++ Compiler Classic](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html#gs.cddfmt) habilita al autovectorizador

```console
user@lab:~ $ icc -o fooO2 foo.c -O2 -qopt-report=3 -xhost -lm
....
LOOP BEGIN at fooO0.c(23,2) inlined into fooO0.c(58,2)
   remark #15300: LOOP WAS VECTORIZED
   remark #15442: entire loop may be executed in remainder
   remark #15450: unmasked unaligned unit stride loads: 1 
   remark #15451: unmasked unaligned unit stride stores: 1 
   remark #15475: --- begin vector cost summary ---
   remark #15476: scalar cost: 111 
   remark #15477: vector cost: 20.750 
   remark #15478: estimated potential speedup: 4.780 
   remark #15482: vectorized math library calls: 1 
   remark #15487: type converts: 2 
   remark #15488: --- end vector cost summary ---
LOOP END
```

## ¿Qué vamos a trabajar?
* Vectorización automática con compilador
* Vectorización semiautomática o guiada (programador)
* Vectorización manual o intrínsecas
* Herramientas de perfilado: Intel-Advisor

### Vectorización automática
* Detección de cuellos de botella
* Eliminación de operaciones de conversiones de datos
* Selección de flags de arquitectura
* Mejora de lay-out memoria

### Vectorización guiada por pragmas
* Las pragmas ayudan al compilador a tomar decisiones

```c
void axpy(float *c, float *a, float *b, float cte, int n)
{
	int i;
	#pragma vector aligned
	for (i=0; i<n; i++)
		c[i] = a[i] + cte*b[i];
}
```

### Vectorización manual con intrínsecas
* [Guía de Intrínsecas](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)

```c
void axpy(float *c, float *a, float *b, float cte, int n)
{
	int i;
	__m128 sse_c, sse_a, sse_b, sse_cte;
	sse_cte = _mm_set1_ps(cte);
	
	for (i=0; i<n; i+=4){
		sse_a = _mm_load_ps(&a[i]);
		sse_b = _mm_load_ps(&a[i]);
		sse_b = _mm_mul_ps(sse_b, sse_cte);
		sse_c = _mm_add_ps(sse_a, sse_b);
		_mm_store_ps(&c[i], sse_c);
	}
}
```


# Ejemplos
## Opciones de compilación 
* Especifica las [optimizaciones en compilación](https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-o): -O[n]
    * Por defecto el compilador utiliza **-O2**
* El código de ejemplo está [disponible](CompilerOpt/foo.c)

|   |   |
|---|---|
| **O0** |  Deshabilita cualquier optimización                                                   |
| **O1** |  Habilita optimizaciones que incrementan tamaño código pero lo acelera                |
|        |  Optimización global: analisis de flujo, análisis de vida de variables                |
|        |  Deshabilita inlining y uso de intrinsecas                                            |
| **O2** | **Activa vectorización**                                                              |
|        | Optimización a nivel de bucle                                                         |
|        | Optimización entre funciones                                                          |
| **O3** | O2+ Opt. más agresiva: fusión de bucles, unroll                                       |
|        | Adecuada para código basados en bucles y cómputo fp                                   |

[^2]: \url{}

## Opciones de compilación
* El código de ejemplo está [disponible](CompilerOpt/foo.c)
    * **O0**: Deshabilita cualquier optimización
    * **qopt-report=3** genera reporte en fichero foo.optprt
    * **-lm** enlaza con la librería matemática

```sh
user@lab:~ $ icc -o fooO0 foo.c -O0 -qopt-report=3 -lm
user@lab:~ $ more foo.optrpt 
Intel(R) Advisor can now assist with vectorization and show optimization
  report messages with your source code.
See "https://software.intel.com/en-us/intel-advisor-xe" for details.

Intel(R) C Intel(R) 64 Compiler Classic for applications running on Intel(R) 64, Version 2021.5.0 Build 20211109_000000
```

    * Modificando la opción de compilación a **O2**: Activa vectorización, en el **reporte indica que en la línea 30** está vectorizado


```sh
user@lab:~ $ icc -o foo02 foo.c -O2 -qopt-report=3 -lm
user@lab:~ $ more foo.optrpt 
....
LOOP BEGIN at foo.c(30,2) inlined into foo.c(60,43)
   remark #15300: LOOP WAS VECTORIZED
   remark #15442: entire loop may be executed in remainder
   remark #15448: unmasked aligned unit stride loads: 1 
   remark #15475: --- begin vector cost summary ---
   remark #15476: scalar cost: 5 
   remark #15477: vector cost: 1.250 
   remark #15478: estimated potential speedup: 3.480 
   remark #15488: --- end vector cost summary ---
LOOP END

....
```

## Cálculo potencial eléctrico
* Cálculo de potencial eléctrico, [código disponible](ElectricPotential/)
    * [Enlace al libro extraido](https://colfaxresearch.com/second-edition-of-parallel-programming-and-optimization-with-intel-xeon-phi-coprocessors/)
    * [Extraido del github!](https://github.com/ColfaxResearch/HOW-Series-Labs/tree/master/4/4.02-vectorization-data-structures-coulomb)

* 4 versiones
    * ver0: versión sin optimizar
        * Conviene prestar atención a las opciones de compilación (añadir arquitectura) y ver ineficiencias (conversiones de datos)
    * ver1: optimizado a la arquitectatura *target*, pero con ineficiencias de acceso a memoria (stride!=1)
        * Propuesta cambiar AoS por SoA
    * ver2: optimización de memoria
    * ver3: optimizacion de memoria a accesos alineados

# Tareas a realizar por el alumno
* Aplicar explotación SIMD utilizando la vectorización automática por parte del compilador en la aplicación de N-Body cuyo [código está disponible en el repositorio](NBody/)
    1. Vectorización autovectorizada
    2. Eliminación de "dependencias" en bucles (desalineación de punteros o \#pragma)
    3. Eliminación de conversión de tipos de datos (float/double)
    4. Accesos a memoria (**¿stride?**): SoA vs AoS
    5. Memoria alineada
* Uso de vectorización guiada porque en [algunos códigos](BlackScholes/) aparecen dependencias de datos aparentes que **inhiben la auyto-vectorización** por parte del compilador
    1. En el directorio "BlackScholes", al compilar con la opción de autovectorizacion (O2) y consultando el report del fichero fuente *black-scholes.c*, se puede ver el mensaje *vector dependence prevents vectorization*
    2. Se recomiendo usar alguna de las *pragma* estudias para "forzar" al compilador a generar código vectorial
```sh
LOOP BEGIN at black-scholes.c(82,5)
   remark #15344: loop was not vectorized: vector dependence prevents vectorization. First dependence is shown below. Use level 5 report for details
   remark #15346: vector dependence: assumed ANTI dependence between s0[i] (84:13) and vput[i] (100:9)
LOOP END
```

* Programación con Intrínsecas haciendo uso de la [Guía de Intrínsecas](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) 
    * El código a utilizar está en el [directorio Intrinsics](Intrinsics/)



