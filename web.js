const tabs = [...document.querySelectorAll('[data-table]')]
const panels = [...document.querySelectorAll('[data-panel]')]

tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(item => {
    const selected = item === tab
    item.classList.toggle('active', selected)
    item.setAttribute('aria-selected', String(selected))
  })
  panels.forEach(panel => panel.classList.toggle('hidden', panel.dataset.panel !== tab.dataset.table))
}))

const dialog = document.querySelector('#chart-dialog')
const dialogImage = dialog.querySelector('img')
document.querySelectorAll('[data-image]').forEach(card => card.addEventListener('click', () => {
  dialogImage.src = card.dataset.image
  dialogImage.alt = card.querySelector('img').alt
  dialog.showModal()
}))
dialog.querySelector('button').addEventListener('click', () => dialog.close())
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close() })
document.querySelector('#year').textContent = new Date().getFullYear()
