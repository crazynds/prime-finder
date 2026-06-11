#pragma once
// time_format.h — formatação de tempo escolhendo automaticamente a melhor unidade
// (min / s / ms / µs / ns). Usado por todos os prints de tempo do projeto.

#include <string>
#include <cstdio>
#include <cmath>

// Formata um intervalo dado em SEGUNDOS, escolhendo a unidade mais legível.
// Troca para a unidade menor só quando o valor cai abaixo de 0,1 dela — assim
// 0,6 ms aparece como "0.600 ms" (e não "600.000 us").
//   >= 60 s    → minutos
//   >= 1 s     → segundos
//   >= 0.1 ms  → milissegundos
//   >= 0.1 µs  → microssegundos
//   senão      → nanossegundos
inline std::string fmt_time(double seconds)
{
    char buf[32];
    double a = std::fabs(seconds);
    if (a >= 60.0)
        snprintf(buf, sizeof(buf), "%.2f min", seconds / 60.0);
    else if (a >= 1.0)
        snprintf(buf, sizeof(buf), "%.3f s", seconds);
    else if (a >= 1e-4)
        snprintf(buf, sizeof(buf), "%.3f ms", seconds * 1e3);
    else if (a >= 1e-7)
        snprintf(buf, sizeof(buf), "%.3f us", seconds * 1e6);
    else
        snprintf(buf, sizeof(buf), "%.0f ns", seconds * 1e9);
    return std::string(buf);
}

// Conveniência para tempos já medidos em MILISSEGUNDOS.
inline std::string fmt_time_ms(double milliseconds)
{
    return fmt_time(milliseconds / 1e3);
}
