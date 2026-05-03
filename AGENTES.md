# **Guía Maestra de Desarrollo para Agentes IA**

Este documento establece las directrices **obligatorias** y universales que deben seguir todos los agentes de IA al generar, refactorizar o analizar código en cualquier proyecto. Es la fuente de verdad para convenciones, arquitectura y estándares de calidad.

---

## **1\. Principios de Código (OBLIGATORIO)**

El código generado debe ser robusto, mantenible y escalable.

### **1.1 SOLID & Clean Code**

* **S** (Single Responsibility): Cada clase, módulo o función debe tener una única responsabilidad.  
* **O** (Open/Closed): El código debe estar abierto para extensión, pero cerrado para modificación.  
* **L** (Liskov Substitution): Las clases derivadas deben poder sustituir a sus clases base sin romper el comportamiento.  
* **I** (Interface Segregation): Preferir interfaces pequeñas y específicas en lugar de interfaces monolíticas.  
* **D** (Dependency Inversion): Depender de abstracciones, no de implementaciones concretas.  
* **DRY (Don't Repeat Yourself)**: Evitar la duplicación. Si un bloque lógico se repite, debe extraerse a una función o utilidad.  
* **KISS (Keep It Simple, Stupid)**: Minimizar la complejidad ciclomática y evitar la sobreingeniería.  
* Preferir siempre la **composición sobre la herencia**.

### **1.2 Naming y Autodocumentación**

* Utilizar nombres descriptivos e intencionales para variables y funciones; el nombre debe explicar qué hace sin necesidad de leer la implementación.  
* Evitar abreviaturas crípticas o de una sola letra (salvo iteradores estándar matemáticos).  
* El código debe ser autodocumentado.

### **1.3 Reglas para Comentarios**

* **Sí:** Explicar decisiones de negocio complejas (el "por qué", no el "qué").  
* **Sí:** Documentar limitaciones de APIs externas o dependencias.  
* **Sí:** Explicar workarounds temporales haciendo referencia al ticket, issue o contexto.  
* **NUNCA:** Comentar código obvio que se explica por sí mismo.  
* **NUNCA:** Dejar bloques de código comentado o muerto.  
* **NUNCA:** Dejar etiquetas "TODO" o "FIXME" sin un contexto claro o ticket asociado.

---

## **2\. Arquitectura y Diseño**

Sin importar si el proyecto usa Domain-Driven Design (DDD), MVC o arquitecturas limpias, se deben respetar los siguientes límites:

* **Aislamiento del Dominio:** La lógica core del negocio o del procesamiento de datos no debe tener dependencias directas de frameworks externos, bases de datos o protocolos de red.  
* **Inyección de Dependencias:** Los servicios externos (bases de datos, clientes HTTP, colas de mensajes) deben inyectarse a través de interfaces o abstracciones.  
* **Gestión de Errores:** No capturar errores genéricamente para ignorarlos. Lanzar errores tipados y específicos del dominio.

---

## **3\. Base de Datos y Persistencia**

Las reglas de seguridad de datos son críticas, especialmente al trabajar con bases de datos relacionales (como PostgreSQL) u ORMs.

* **NUNCA** modificar un archivo de migración que ya ha sido aplicado o comiteado.  
* **NUNCA** sugerir ni ejecutar comandos destructivos (DROP TABLE, DROP COLUMN, TRUNCATE, DELETE FROM sin WHERE) sin pedir confirmación explícita primero.  
* Utilizar cláusulas de seguridad (IF NOT EXISTS / IF EXISTS) siempre que el motor SQL lo permita.  
* Toda nueva columna no anulable debe tener un valor DEFAULT lógico.  
* Las operaciones de escritura múltiple que dependan unas de otras deben envolverse siempre en **Transacciones** para asegurar la atomicidad.

---

## **4\. Estrategia de Testing**

El código generado debe ser testeable por diseño. Si se solicita crear pruebas, seguir estas reglas:

* **Patrón AAA:** Separar visualmente las fases de Arrange (Preparar), Act (Actuar) y Assert (Afirmar).  
* **Aislamiento:** Los tests deben ser independientes entre sí y no compartir estado mutable.  
* **Mocks:** Burlar (mockear) todas las dependencias externas (I/O, bases de datos, red) en las pruebas unitarias.  
* **Un Assert por concepto:** Limitar las aserciones a un único comportamiento o concepto por prueba.  
* **Naming:** Utilizar convenciones claras, por ejemplo: should\[ComportamientoEsperado\]When\[Condicion\].

---

## **5\. Control de Versiones y Git**

Si el agente asiste en la creación de commits o gestión de ramas:

* Usar **Conventional Commits** para los mensajes.  
  * *Tipos permitidos:* feat, fix, refactor, test, docs, chore, ci, perf.  
  * *Ejemplo:* feat(auth): implementar validación de tokens JWT.  
* Cada Pull Request/Commit importante debe enfocarse en una sola responsabilidad lógica.

---

## **6\. Flujo de Trabajo y Pre-Commit (Mentalidad)**

Antes de dar por completada una respuesta o tarea, el agente debe verificar internamente:

* ¿El código pasaría un linter estándar sin errores?  
* ¿Se han considerado los casos límite (edge cases) y valores nulos?  
* ¿Existen credenciales, tokens o información sensible hardcodeada? (De ser así, extraer a variables de entorno).

---

## **7\. Reglas Ejecutivas para el Agente IA**

### **SIEMPRE debes:**

* Tipar fuertemente el código (ej. TypeScript strict mode, Type Hints en Python). Evitar el uso de tipos dinámicos o any a menos que sea estrictamente necesario.  
* Verificar las importaciones para asegurar que no se violen las capas arquitectónicas (ej. no importar infraestructura desde la presentación).  
* Proporcionar la solución más simple que funcione de manera segura y eficiente.

### **NUNCA debes:**

* Inventar funciones de librerías de terceros que no existen en la documentación oficial. Si no estás seguro, indícalo.  
* Ignorar vulnerabilidades de seguridad comunes (OWASP Top 10\) al estructurar APIs o procesar datos.  
* Crear archivos adicionales innecesarios si la lógica puede vivir orgánicamente en la estructura actual.

