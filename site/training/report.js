'use strict';
const xy = document.querySelector('#xy-error');
const z = document.querySelector('#z-error');
function updateKernel() {
  const horizontal = Number(xy.value), vertical = Number(z.value);
  const rate = 0.5 * Math.exp(-((horizontal / 0.3) ** 2 + (vertical / 0.45) ** 2));
  document.querySelector('#xy-label').textContent = `${horizontal.toFixed(2)} m`;
  document.querySelector('#z-label').textContent = `${vertical.toFixed(2)} m`;
  document.querySelector('#kernel-rate').textContent = rate.toFixed(6);
  document.querySelector('#kernel-step').textContent = (0.02 * rate).toFixed(6);
}
xy.addEventListener('input', updateKernel);
z.addEventListener('input', updateKernel);
updateKernel();
document.querySelector('#print-report').addEventListener('click', () => window.print());
if (window.matchMedia('(max-width:750px)').matches) {
  document.querySelector('.contents details').open = false;
}
